#
import numpy as np
import gym
from gym import spaces
from torch.utils.data import DataLoader
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
        obs_raw = self.market_iter.next() # 怎样获得当天收盘价？
        print('X: {0}; y: {1}; '.format(obs_raw[0].shape, obs_raw[1].shape))

    def learn(self):
        obs = self.observation_space.sample()
        print(obs)