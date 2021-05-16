#
import math
import unittest
from fak.option.constant_short_rate import ConstantShortRate

class TConstantShortRate(unittest.TestCase):
    def test_poc(self):
        r = -0.05
        d0 = 100
        t = 1.0 # 一年后
        d0_t = math.exp(-r*t)
        dt = d0_t * d0
        print('折现率：{0}; 价值：{1};'.format(d0_t, dt))