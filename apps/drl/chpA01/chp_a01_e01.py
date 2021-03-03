#
import numpy as np

class ChpA01E01(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print('线性回归')
        X, y = self.load_ds()
        print('X: {0};'.format(X))
        print('y: {0};'.format(y))

    def load_ds(self):
        b = 1.5
        w = 2.3
        X = np.linspace(1, 10, num=10)
        y = w*X + b
        return X, y