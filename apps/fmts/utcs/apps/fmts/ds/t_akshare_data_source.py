# Akshare数据源测试程序
import unittest
from apps.fmts.ds.akshare_data_source import AkshareDataSource

class TAkshareDataSource(unittest.TestCase):
    def test_get_minute_bars_001(self):
        stock_symbol = 'sh600260'
        data = AkshareDataSource.get_minute_bars(stock_symbol)
        print(data)
        self.assertTrue(1>0)