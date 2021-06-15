#
import numpy as np
import torch
import torch.nn as nn

class A2cModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(A2cModel, self).__init__()
        print('input_shape: {0}; {1};'.format(input_shape, input_shape[0]))
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        print('conv_out_size: {0};'.format(conv_out_size))
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        print('_get_conv_out: shape: {0};'.format(shape))
        v1 = torch.zeros(1, *shape)
        print('_get_conv_out: v1: {0};'.format(v1.shape))
        ####
        o = self.conv(torch.zeros(1, *shape))
        print('_get_conv_out: o: {0};'.format(o.shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)