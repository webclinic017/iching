#
import numpy as np
import gym
from gym import spaces
from torch.utils.data import DataLoader
from biz.dmrl.app_config import AppConfig
from biz.dmrl.market import Market
from apps.fmts.gui.fmts_human_render import FmtsHumanRender
from apps.fmts.gui.fmts_text_render import FmtsTextRender

class FmtsEnv(gym.Env):
    RENDER_MODE_TEXT = 'text'
    RENDER_MODE_HUMAN = 'human'

    def __init__(self, stock_symbol, mode='human'):
        super(FmtsEnv, self).__init__()
        self.stock_symbol = stock_symbol
        self.current_step = 0
        self.trade_mode = AppConfig.TRADE_MODE_HOLD
        if FmtsEnv.RENDER_MODE_TEXT == mode:
            self.renderer = FmtsTextRender()
        elif FmtsEnv.RENDER_MODE_HUMAN == mode:
            self.renderer = FmtsHumanRender()
        else:
            self.renderer = FmtsTextRender()
        self.window_size = 50 # 绘制50个交易日的图像
        self.trades = {}
        # 初始化交易历史信息
        self.trades['current_step'] = 1
        self.trades['balance'] = 0.0
        self.trades['position'] = 0
        self.trades['price'] = 0.0
        self.trades['quant'] = 0
        self.trades['net_value'] = 0.0
        self.trades['trade_mode'] = AppConfig.TRADE_MODE_HOLD
        self.trades['trade_dates'] = []
        self.trades['balances'] = []
        self.trades['positions'] = []
        self.trades['prices'] = []
        self.trades['net_values'] = []
        self.trades['bars'] = {}
        self.trades['bars']['dates'] = []
        self.trades['bars']['Open'] = []
        self.trades['bars']['High'] = []
        self.trades['bars']['Low'] = []
        self.trades['bars']['Close'] = []
        self.trades['bars']['Volume'] = []
        self.trades['trade_history'] = []
        self.market = Market(stock_symbol)
        # self.aks_iter.next() raise StopIteration exception when end
        self.batch_size = 1
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        # 现金、仓位、净值（可以由收盘价*仓位+现金求出）
        # 价格按对数收益率表示，交易量按(x-mu)/std，这些值基本都在-1.0到1.0之间
        self.observation_space = spaces.Box(
            low=-10000.0, high=10000.0, shape=(50, 8), dtype=np.float16)

    def reset(self):
        market_loader = DataLoader(
            self.market,
            batch_size = self.batch_size,
            num_workers = 0,
            shuffle = True,
            drop_last = True
        )
        self.market_iter = iter(market_loader)
        self.balance = AppConfig.rl_env_params['initial_balance']
        self.position = AppConfig.rl_env_params['initial_position']
        self.net_value = 0.0
        self.current_step = 1
        self.trade_mode = AppConfig.TRADE_MODE_HOLD
        self.price = 0.0 # 交易价格
        self.quant = 0 # 产易量
        self.obs = self._next_obs()
        # 初始化交易历史信息
        self.trades = {}
        self.trades['current_step'] = 1
        self.trades['balance'] = 0.0
        self.trades['position'] = 0
        self.trades['price'] = 0.0
        self.trades['quant'] = 0
        self.trades['net_value'] = 0.0
        self.trades['trade_mode'] = AppConfig.TRADE_MODE_HOLD
        self.trades['trade_dates'] = []
        self.trades['balances'] = []
        self.trades['positions'] = []
        self.trades['prices'] = []
        self.trades['net_values'] = []
        self.trades['bars'] = {}
        self.trades['bars']['dates'] = []
        self.trades['bars']['Open'] = []
        self.trades['bars']['High'] = []
        self.trades['bars']['Low'] = []
        self.trades['bars']['Close'] = []
        self.trades['bars']['Volume'] = []
        self.trades['trade_history'] = []
        # 绘制图像
        self.render()
        return self.obs

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        self.obs = self._next_obs()
        reward = 1.0
        if self.obs is None:
            done = True
        else:
            done = False
            self.render()
        info = {}
        return self.obs, reward, done, info

    def render(self):
        '''
        显示交易信息，mode=text时将信息打印到后台窗口；mode=human以Matplotlib动画方式显示
        '''
        self.trades['window_size'] = self.window_size
        self.trades['current_step'] = self.current_step
        self.trades['balance'] = self.balance
        self.trades['position'] = self.position
        self.trades['price'] = self.price
        self.trades['quant'] = self.quant
        self.trades['net_value'] = self.net_value
        self.trades['trade_mode'] = self.trade_mode
        idxs_to_del = []
        th_len = len(self.trades['trade_history'])
        for idx in range(th_len):
            self.trades['trade_history'][idx]['idx'] -= 1
            if self.trades['trade_history'][idx]['idx'] < 0:
                idxs_to_del.append(idx)
        for idx in idxs_to_del:
            self.trades['trade_history'].pop(idx)
        if self.trade_mode != AppConfig.TRADE_MODE_HOLD:
            item = {}
            item['idx'] = len(self.trades['trade_dates']) - 1
            item['price'] = self.price
            item['quant'] = self.quant
            item['trade_mode'] = self.trade_mode
            self.trades['trade_history'].append(item)
        # 历史信息
        trade_date = self.market.get_trade_date(self.current_step)[0]
        self.trades['trade_dates'] = self.shift_append(self.window_size, self.trades['trade_dates'], trade_date)
        self.trades['balances'] = self.shift_append(self.window_size, self.trades['balances'], self.balance)
        self.trades['positions'] = self.shift_append(self.window_size, self.trades['positions'], self.price)
        self.trades['prices'] = self.shift_append(self.window_size, self.trades['prices'], self.price)
        self.trades['net_values'] = self.shift_append(self.window_size, self.trades['net_values'], self.net_value)
        # 添加K线图所需数据
        self.trades['bars']['dates'] = self.shift_append(self.window_size, self.trades['bars']['dates'], trade_date)
        self.trades['bars']['Open'] = self.shift_append(self.window_size, self.trades['bars']['Open'], self.obs[0][50])
        self.trades['bars']['High'] = self.shift_append(self.window_size, self.trades['bars']['High'], self.obs[0][51])
        self.trades['bars']['Low'] = self.shift_append(self.window_size, self.trades['bars']['Low'], self.obs[0][52])
        self.trades['bars']['Close'] = self.shift_append(self.window_size, self.trades['bars']['Close'], self.obs[0][53])
        self.trades['bars']['Volume'] = self.shift_append(self.window_size, self.trades['bars']['Volume'], self.obs[0][54])
        # 其他环境信息
        self.trades['obs'] = self.obs
        self.renderer.render(self.trades)


    def _next_obs(self):
        try:
            obs = self.market_iter.next()
            obs[0] = np.append(obs[0], self.balance)
            obs[0] = np.append(obs[0], self.position)
            obs[0] = np.append(obs[0], self.net_value)
            obs[0] = np.append(obs[0], obs[1].item())
        except:
            obs = None
        return obs

    def _take_action(self, action):
        action_type = action[0]
        action_percent = action[1]
        price = self.obs[0][53] # 取出收盘价
        #balance = obs[54]
        if action_type < 1:
            # 买入股票
            buy_total = int(self.balance / price)
            buy_amount = int(buy_total * action_percent)
            delta = 0
            if buy_amount * price*(1+AppConfig.rl_env_params['buy_commission_rate']) > self.balance:
                for delta in range(buy_amount):
                    if (buy_amount-delta)*price*(1+AppConfig.rl_env_params['buy_commission_rate']) < self.balance:
                        break
            buy_amount -= delta
            if buy_amount < 10:
                self.trade_mode = AppConfig.TRADE_MODE_HOLD
                self.price = 0.0
                self.quant = 0
                print('    持有（忽略买入指令）')
            else:
                raw_amount = buy_amount * price
                commission = raw_amount * AppConfig.rl_env_params['buy_commission_rate']
                amount = raw_amount + commission
                # 更新账户信息
                self.balance -= amount
                self.position += buy_amount
                self.net_value = self.balance + self.position * price
                self.trade_mode = AppConfig.TRADE_MODE_BUY
                self.price = price
                self.quant = buy_amount
        elif action_type < 2:
            sell_amount = int(self.position * action_percent)
            if sell_amount < 10:
                self.trade_mode = AppConfig.TRADE_MODE_HOLD
                self.price = 0.0
                self.quant = 0
                print('    持有（忽略卖出指令）')
            else:
                raw_amount = sell_amount * price
                commission = raw_amount * AppConfig.rl_env_params['sell_commission_rate']
                amount = raw_amount - commission
                # 更新账户信息
                self.balance += amount
                self.position -= sell_amount
                self.net_value = self.balance + self.position * price
                self.trade_mode = AppConfig.TRADE_MODE_SELL
                self.price = price
                self.quant = sell_amount
        else:
            self.trade_mode = AppConfig.TRADE_MODE_HOLD
            self.price = 0.0
            self.quant = 0


    

    def shift_append(self, window_size, list_obj, val):
        '''
        如果列表长度小于window_size时，直接将val添加到列表最后，如果长度大于等于window_size时，则
        去掉列表第一个元素，然后将val添加到列表最后
        '''
        if len(list_obj) < window_size:
            list_obj.append(val)
        else:
            list_obj = list_obj[1:]
            list_obj.append(val)
        return list_obj

    










    def learn(self):
        obs = self.observation_space.sample()
        print(obs)