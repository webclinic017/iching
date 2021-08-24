#
import unittest
from apps.fmts.ds.akshare_data_source import AkshareDataSource
from apps.fmts.ds.ohlcv_processor import OhlcvProcessor

class TOhlcvProcessor(unittest.TestCase):
    def test_draw_close_price_curve_001(self):
        stock_symbol = 'sh600260'
        OhlcvProcessor.draw_close_price_curve(stock_symbol, mode=OhlcvProcessor.PCM_TICK)
        
    def test_draw_close_price_curve_002(self):
        stock_symbol = 'sh600260'
        OhlcvProcessor.draw_close_price_curve(stock_symbol, mode=OhlcvProcessor.PCM_DATETIME)

    def test_gen_1d_log_diff_norm_001(self):
        stock_symbol = 'sh600260'
        items = AkshareDataSource.get_minute_bars(stock_symbol=stock_symbol)
        OhlcvProcessor.gen_1d_log_diff_norm(stock_symbol, items)

    def test_get_ds_raw_data(self):
        stock_symbol = 'sh600260'
        X, y, info = OhlcvProcessor.get_ds_raw_data(stock_symbol, window_size=10)
        print('X: {0};'.format(X.shape))
        print('y: {0};'.format(y))
        print('info: {0};'.format(info))