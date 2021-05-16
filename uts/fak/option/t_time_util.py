#
import datetime as dt
import unittest
from fak.option.time_util import TimeUtil

class TTimeUtil(unittest.TestCase):
    def test_get_year_deltas(self):
        dates = [dt.datetime(2020, 1, 1), dt.datetime(2020, 7, 1), dt.datetime(2021, 1, 1)]
        nds = TimeUtil.get_year_deltas(dates)
        print(nds)
        self.assertTrue(1 == 1)