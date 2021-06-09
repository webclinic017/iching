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
from biz.drlt.rll.experience import ExperienceSourceFirstLast

class TExperienceSourceFirstLast(unittest.TestCase):
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
        agent = DQNAgent(net, selector, device=device)
        exp_source = rll.experience.ExperienceSourceFirstLast(
            env, agent, AppConfig.GAMMA, steps_count=AppConfig.REWARD_STEPS)
        src_itr = iter(exp_source)
        v1 = next(src_itr)
        print('v1: {0}; {1};'.format(type(v1), v1))