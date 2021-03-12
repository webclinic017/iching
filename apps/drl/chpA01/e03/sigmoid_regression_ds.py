# 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class SigmoidRegressionDs(object):
    X_MIN = -15
    X_M1 = 1
    X_M2 = 3
    X_MAX = 19
    Y_MIN = 1.0
    Y_MAX = 2.0
    LEFT_PTS = 100
    MID_PTS = 100
    RIGHT_PTS = 100

    def __init__(self):
        self.name = ''
        X1 = np.linspace(SigmoidRegressionDs.X_MIN, SigmoidRegressionDs.X_M1, SigmoidRegressionDs.LEFT_PTS)
        y1 = SigmoidRegressionDs.Y_MIN * np.ones_like(X1)
        X2 = np.linspace(SigmoidRegressionDs.X_M1, SigmoidRegressionDs.X_M2, SigmoidRegressionDs.MID_PTS)
        y2 = 0.5*X2 + 0.5
        X3 = np.linspace(SigmoidRegressionDs.X_M2, SigmoidRegressionDs.X_MAX, SigmoidRegressionDs.RIGHT_PTS)
        y3 = SigmoidRegressionDs.Y_MAX*np.ones_like(X3)
        self.X = np.append(X1, [X2, X3])
        self.y = np.append(y1, [y2, y3])

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