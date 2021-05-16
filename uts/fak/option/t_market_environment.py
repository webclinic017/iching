#
import datetime as dt
import unittest
from fak.option.market_environment import MarketEnvironment

class TMarketEnvironment(unittest.TestCase):
    def test_add_constant_001(self):
        me1 = MarketEnvironment('me1', dt.datetime(2020, 1, 1))
        me1.add_constant('csr', 3.6)
        print('csr={0};'.format(me1.get_constant('csr')))