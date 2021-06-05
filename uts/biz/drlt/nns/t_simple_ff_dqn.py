# 
import unittest
import numpy as np
import torch
from biz.drlt.app_config import AppConfig
from biz.drlt.nns.simple_ff_dqn import SimpleFFDQN
from biz.drlt.ds.bar_data import BarData
from biz.drlt.envs.minute_bar_env import MinuteBarEnv

class TSimpleFfDqn(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def test_forward(self):
        device = torch.device("cuda:0")
        year = 2016
        instrument = 'data\\YNDX_160101_161231.csv'
        stock_data = BarData.load_year_data(year)
        print('stock_data: {0};'.format(stock_data[instrument]))
        env = MinuteBarEnv(
                stock_data, bars_count=AppConfig.BARS_COUNT, volumes=True)
        print('obs_len={0}; actions_n={1};'.format(env.observation_space.shape[0],
                                env.action_space.n))
        net = SimpleFFDQN(env.observation_space.shape[0],
                                env.action_space.n).to(device)
        mu = 1.0
        sigma = 3.0
        simples = 42
        x = torch.from_numpy(np.random.normal(mu, sigma, simples).reshape(1, 42)).to(device=device, dtype=torch.float32)
        y = net(x)
        print('y: {0}; {1}; {2};'.format(type(y), y.shape, y))