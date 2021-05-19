#
import math
from threading import Thread
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# 
from apps.drl.c03.e03.iching_tb import IchingTb

class SinEnv(gym.Env):
    MAX_STEPS = 1000
    RENDER_PER_STEPS = 10

    def __init__(self):
        self.step_left = SinEnv.MAX_STEPS
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(1)
        self.tb = IchingTb()

    def step(self, action):
        done = self.step_left <= 0
        reward = 0.0
        v = math.sin(360/SinEnv.MAX_STEPS * (SinEnv.MAX_STEPS - self.step_left) / 180.0 * math.pi)
        self.step_left -= 1
        self.obs = np.array([v])
        return self.obs, reward, done, {}

    def reset(self) -> np.array:
        self.step_left = SinEnv.MAX_STEPS
        self.obs = np.array([0.0])
        self.tb.reset_plot()
        return self.obs

    def render(self, mode='human'):
        print('step left: {0}; v={1};'.format(self.step_left, self.obs[0]))
        if self.step_left % SinEnv.RENDER_PER_STEPS == 0:
            self.tb.update_plot(SinEnv.MAX_STEPS - self.step_left, self.obs[0])

    def close(self):
        self.tb.stop_plot()
        return None