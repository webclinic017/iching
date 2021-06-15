# 
import unittest
from biz.drlt.nns.a2c_model import A2cModel

class TA2cModel(unittest.TestCase):
    def test_exp(self):
        obs_space_shape = (4, 84, 84)
        actions_n = 6
        net = A2cModel(obs_space_shape, actions_n)
        self.assertTrue(1>0)