# 
import numpy as np
from scipy import linalg

class C02E01(object):
    idx = 0
    ts = np.array([[0.5, 0.8, 0.4], [1.8, 0.3, 0.3], [-2.2, -1.3, 3.5]])

    def __init__(self):
        self.name = 'apps.mml.c02.c02_e01.C02E01'

    def startup(self, args={}):
        np.random.seed(1000)
        B1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        B2 = self.generate_base_from_base(B1)
        print('base2: {0};'.format(B2))
        x = np.array([1.1, 2.4, 3.5])
        x_c_1 = np.linalg.solve(B1, x)
        print('x_c_1: {0};'.format(x_c_1))
        x_c_2 = np.linalg.solve(B2, x)
        print('x_c_2: {0};'.format(x_c_2))

    def vector_linear_combination(self, base_v, low=-10.0, high=10.0):
        i_debug = 1
        if 1 == i_debug:
            t = C02E01.ts[C02E01.idx]
            C02E01.idx += 1
        else:
            t = np.random.uniform(low, high, (base_v.shape[1],))
        v = t[0] * base_v[0] + t[1] * base_v[1] + t[2] * base_v[2]
        return v

    def generate_base_from_base(self, base_v):
        b_0 = self.vector_linear_combination(base_v)
        b_1 = self.vector_linear_combination(base_v)
        b_2 = self.vector_linear_combination(base_v)
        return np.array([b_0, b_1, b_2])