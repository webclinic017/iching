# 
import torch
import torch.nn as nn

class ChpA01E01Model(nn.Module):
    def __init__(self):
        super(ChpA01E01Model, self).__init__()
        w = torch.tensor(1.0, requires_grad=True)
        self.register_parameter('w001', torch.nn.Parameter(data=w, requires_grad=True))
        b = torch.tensor(0.0, requires_grad=True)
        self.register_parameter('b001', torch.nn.Parameter(data=b, requires_grad=True))

    def forward(self, X):
        return self.w001 * X + self.b001