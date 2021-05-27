# 二维空间旋转45度
import math
import numpy as np

class C02E02(object):
    def __init__(self):
        self.name = 'apps.mml.c02.c02_e02.C02E02'

    def startup(self, args={}):
        print('二维空间坐标变换')
        b1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        A_phi = np.array([[math.cos(math.pi / 4.0), -math.sin(math.pi / 4.0)], [math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]])
        A_phi_i = 1 / (A_phi[0][0]*A_phi[1][1] - A_phi[0][1]*A_phi[1][0]) * np.array([[A_phi[1][1], -A_phi[0][1]], [-A_phi[1][0], A_phi[0][0]]])
        b2 = np.matmul(b1, A_phi_i)
        print('b2: {0};'.format(b2))
        x_1_hat = np.array([1.0, 1.0])
        y_1_hat = np.matmul(b2, x_1_hat.T)
        print('y_1_hat={0};'.format(y_1_hat))