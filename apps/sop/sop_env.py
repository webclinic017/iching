# 
import numpy as np
import gym
from gym import spaces
# 
from apps.sop.sop_agent import SopAgent
from apps.sop.ds.sh50etf_dataset import Sh50etfDataset

class SopEnv(gym.Env):
    def __init__(self):
        self.refl = ''
        self.tick = 0
        # action为3维数组：1维-是期权合约编号；2维-买入持有卖出；
        # 3维-百分比，缺省为100%
        self.agent = SopAgent()

    def startup(self, args={}):
        self.ds = Sh50etfDataset()
        self.reset()
        obs, reward, done, info = self._next_observation(), 0, False, {}
        for dt in self.ds.dates:
            print('{0}: '.format(dt))
            action = self.agent.choose_action(obs, reward)
            obs, reward, done, info = self.step(action)
            X = obs['X'].cpu().numpy()
            y = obs['y'].cpu().numpy()
            r = obs['r'].cpu().numpy()
            self.tick += 1

    def reset(self):
        print('重置环境到初始状态')
        self.agent.reset(self)
        self.tick = 0

    def _next_observation(self):
        X, y, r = self.ds.__getitem__(self.tick)
        return {'X': X, 'y': y, 'r': r}

    def step(self, action):
        self._take_action(action)
        obs = self._next_observation()
        reward = 0.0
        done = False
        return obs, reward, done, {}

    def _take_action(self, action):
        pass