#
import torch
import torch.nn as nn

class SigmoidRegressionModel(nn.Module):
    def __init__(self):
        super(SigmoidRegressionModel, self).__init__()
        w = torch.tensor(1.0, requires_grad=True)
        self.register_parameter('w001', torch.nn.Parameter(data=w, requires_grad=True))
        b = torch.tensor(0.0, requires_grad=True)
        self.register_parameter('b001', torch.nn.Parameter(data=b, requires_grad=True))
        c = torch.tensor(1.0, requires_grad=True)
        self.register_parameter('c001', torch.nn.Parameter(data=c, requires_grad=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        return self.c001 * self.sigmoid(self.w001 * X + self.b001)