#
import numpy as np
import gym
from gym import spaces

class AksEnv(gym.Env):
    def __init__(self):
        super(AksEnv, self).__init__()
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        # 现金、仓位，因为净值可以由收盘价*仓位+现金求出，所以不包括在内
        # 价格按对数收益率表示，交易量按(x-mu)/std，这些值基本都在-1.0到1.0之间
        self.observation_space = spaces.Box(
            low=-10000.0, high=10000.0, shape=(50, 7), dtype=np.float16)

    def learn(self):
        obs = self.observation_space.sample()
        print(obs)