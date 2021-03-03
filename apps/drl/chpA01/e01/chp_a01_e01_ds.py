# 
import numpy as np
from torch.utils.data import Dataset

class ChpA01E01Ds(Dataset):
    def __init__(self, num):
        b = 1.6
        w = 0.3
        self.X = np.linspace(0, 1.0, num=num)
        self.y = w*self.X + b

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)