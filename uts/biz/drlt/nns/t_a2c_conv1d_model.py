# 
import unittest
from biz.drlt.nns.a2c_conv1d_model import A2cConv1dModel

class TA2cConv1dModel(unittest.TestCase):
    def test_exp(self):
        obs_n = 42 # obs = np.zeros(1, 42)
        action_n = 3
        net = A2cConv1dModel((1, obs_n), action_n)
        print('A2cConv1dModel is OK')