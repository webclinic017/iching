#
import numpy as np
import gym
from gym import spaces

class AksEnv(gym.Env):
    def __init__(self):
        super(AksEnv, self).__init__()
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

    def learn(self):
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        action = self.action_space.sample()
        print('行动：{0};'.format(action))
        if action[0] < 1:
            print('买入：{0}%;'.format(action[1]*100))
        elif action[0] < 2:
            print('卖出：{0}%;'.format(action[1]*100))
        else:
            print('持有...')