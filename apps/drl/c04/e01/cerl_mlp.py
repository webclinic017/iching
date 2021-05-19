#
import torch
import torch.nn as nn

class CerlMlp(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(CerlMlp, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)