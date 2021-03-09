#
import torch
import torch.nn as nn

class Covid2019Model(nn.Module):
    def __init__(self, input_dim):
        super(Covid2019Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        return self.net(X).squeeze(1)