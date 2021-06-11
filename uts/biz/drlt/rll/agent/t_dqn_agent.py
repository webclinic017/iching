# 
import torch
import gym
import unittest
import biz.drlt.rll as rll
from biz.drlt.app_config import AppConfig
from biz.drlt.ds.bar_data import BarData
from biz.drlt.envs.minute_bar_env import MinuteBarEnv
from biz.drlt.nns.simple_ff_dqn import SimpleFFDQN
from biz.drlt.rll.agent import DQNAgent

class TDqnAgent(unittest.TestCase):
    def test_exp(self):
        #
        device = torch.device("cuda:0")
        year = 2016
        stock_data = BarData.load_year_data(year)
        env = MinuteBarEnv(
                stock_data, bars_count=AppConfig.BARS_COUNT)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        net = SimpleFFDQN(env.observation_space.shape[0],
                                env.action_space.n).to(device)
        selector = rll.actions.EpsilonGreedyActionSelector(AppConfig.EPS_START)
        agt = DQNAgent(net, selector, device=device)
        obs = env.reset()
        '''
        actions, agent_states = agt(obs)
        print('actions: {0}; {1};'.format(type(actions), actions))
        print('action_states: {0}; {1};'.format(type(agent_states), agent_states))
        '''
