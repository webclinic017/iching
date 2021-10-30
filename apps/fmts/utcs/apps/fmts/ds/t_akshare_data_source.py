# Akshare数据源测试程序
import unittest
from apps.fmts.ds.akshare_data_source import AkshareDataSource

class TAkshareDataSource(unittest.TestCase):
    def test_get_minute_bars_001(self):
        '''
        测试正常情况，运行方式：
        python -m unittest apps.fmts.utcs.apps.fmts.ds.t_akshare_data_source.TAkshareDataSource.test_get_minute_bars_001 -v
        '''
        stock_symbol = 'sh600260'
        data = AkshareDataSource.get_minute_bars(stock_symbol)
        for item in data:
            print(item)
        self.assertTrue(1>0)