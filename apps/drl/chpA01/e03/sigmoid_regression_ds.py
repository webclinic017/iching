# 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class SigmoidRegressionDs(object):
    def __init__(self):
        self.name = ''
        X1 = np.linspace(-5, 1, 100)
        y1 = np.ones_like(X1)
        X2 = np.linspace(1, 3, 100)
        y2 = 0.5*X2 + 0.5
        X3 = np.linspace(3, 9, 100)
        y3 = 2.0*np.ones_like(X3)
        self.X = np.append(X1, [X2, X3])
        self.y = np.append(y1, [y2, y3])
        self.draw_c1_hard_sigmoid(self.X, self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def draw_c1_hard_sigmoid(self, X, y):
        plt.figure(figsize=(5, 5))
        plt.plot(X, y, c='r')
        plt.xlim(-5.5, 9.5)
        plt.ylim(0.0, 3.5)
        plt.show()