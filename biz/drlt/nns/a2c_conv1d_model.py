# A2C算法1维卷积模型
import numpy as np
import torch
import torch.nn as nn

class A2cConv1dModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(A2cConv1dModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        # 策略网络定义
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        # 值网络定义
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1])
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)