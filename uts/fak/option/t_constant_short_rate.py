#
from fak.option.time_util import TimeUtil
import math
import datetime as dt
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

    def test_get_discount_factors_obj(self):
        dates = [dt.datetime(2020, 1, 1), dt.datetime(2020, 7, 1), dt.datetime(2021, 1, 1)]
        csr = ConstantShortRate('csr', 0.05)
        f = csr.get_discount_factors(dates)
        print('discount_factor: {0};'.format(f))

    def test_get_discount_factors_val(self):
        dates = [dt.datetime(2020, 1, 1), dt.datetime(2020, 7, 1), dt.datetime(2021, 1, 1)]
        csr = ConstantShortRate('csr', 0.05)
        dv = TimeUtil.get_year_deltas(dates)
        f = csr.get_discount_factors(dv, dtobjects=False)
        print('discount_factor: {0};'.format(f))