# 
import numpy as np
import torch
import torch.nn as nn
import unittest
from biz.drlt.nns.a2c_conv1d_model import A2cConv1dModel

class TA2cConv1dModel(unittest.TestCase):
    def test_exp(self):
        obs_n = 42 # obs = np.zeros(1, 42)
        action_n = 3
        net = A2cConv1dModel((1, obs_n), action_n)
        # prepare input
        mu = 0.0
        std = 1.0
        x_raw = np.random.normal(mu, std, obs_n)
        x = torch.from_numpy(x_raw.reshape((1, 42))).float()
        print('x: {0};'.format(x.shape))
        actions, v_pi = net(x)
        print('actions: {0}; {1};'.format(actions.shape, actions))
        print('v_pi: {0};'.format(v_pi))

    def __t001(self):
        obs_n = 42 # obs = np.zeros(1, 42)
        action_n = 3
        net = A2cConv1dModel((1, obs_n), action_n)
        # prepare input
        mu = 0.0
        std = 1.0
        x_raw = np.random.normal(mu, std, obs_n)
        x = torch.from_numpy(x_raw.reshape((1, 1, 42))).float()
        print('x: {0};'.format(x.shape))
        # define first layer
        l1 = nn.Conv1d(1, 32, kernel_size=3, stride=1)
        a1 = l1(x)
        print('a1: {0}; '.format(a1.shape))
        # define second layer
        l2 = nn.ReLU()
        a2 = l2(a1)
        print('a2: {0};'.format(a2.shape))
        # define the third layer
        l3 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        a3 = l3(a2)
        print('a3: {0};'.format(a3.shape))
        # define the forth layer
        l4 = nn.ReLU()
        a4 = l4(a3)
        print('a4: {0};'.format(a4.shape))
        # define the fifth layer
        l5 = nn.Conv1d(64, 64, kernel_size=3, stride=1)
        a5 = l5(a4)
        print('a5: {0};'.format(a5.shape))
        # define the sixth layer
        l6 = nn.ReLU()
        a6 = l6(a5)
        print('a6: {0};'.format(a6.shape))
        print('A2cConv1dModel is OK')