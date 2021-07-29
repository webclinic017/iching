#
import numpy as np
import gym
from gym import spaces
from torch.utils.data import DataLoader
from biz.dmrl.app_config import AppConfig
from biz.dmrl.market import Market

class AksEnv(gym.Env):
    def __init__(self, stock_symbol):
        super(AksEnv, self).__init__()
        self.stock_symbol = stock_symbol
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
        self.obs = self._next_obs()
        self.render()
        return self.obs

    def step(self, action):
        self._take_action(action)
        self.obs = self._next_obs()
        reward = 1.0
        if self.obs is None:
            done = True
        else:
            done = False
            self.render()
        info = {}
        return self.obs, reward, done, info

    def render(self, mode='human'):
        print('{0}：bar({1}, {2}, {3}, {4}), state=(余额：{5}, 仓位：{6}, 净值：{7})'.format(
            self.current_step, self.obs[0][50], self.obs[0][51], self.obs[0][52], self.obs[0][53],
            self.balance, self.position, self.net_value
        ))


    def _next_obs(self):
        try:
            obs = self.market_iter.next()
            obs[0] = np.append(obs[0], self.balance)
            obs[0] = np.append(obs[0], self.position)
            obs[0] = np.append(obs[0], self.net_value)
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
            raw_amount = buy_amount * price
            commission = raw_amount * AppConfig.rl_env_params['buy_commission_rate']
            amount = raw_amount + commission
            # 更新账户信息
            self.balance -= amount
            self.position += buy_amount
            self.net_value = self.balance + self.position * price
            print('    买入：数量：{0}; 金额：{1}；余额：{2}；仓位：{3}；净值：{4}'.format(
                buy_amount, amount, self.balance, self.position, self.net_value
            ))
        elif action_type < 2:
            sell_amount = int(self.position * action_percent)
            raw_amount = sell_amount * price
            commission = raw_amount * AppConfig.rl_env_params['sell_commission_rate']
            amount = raw_amount - commission
            # 更新账户信息
            self.balance += amount
            self.position -= sell_amount
            self.net_value = self.balance + self.position * price
            print('    卖出：数量：{0}；金额：{1}；余额：{2}；仓位：{3}；净值：{4}；'.format(
                sell_amount, amount, self.balance, self.position, self.net_value
            ))
        else:
            print('    持有')

    










    def learn(self):
        obs = self.observation_space.sample()
        print(obs)