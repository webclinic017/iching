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
        # 准备全量数据集
        items = AkshareDataSource.get_minute_bars(stock_symbol=stock_symbol)
        OhlcvProcessor.gen_1d_log_diff_norm(stock_symbol, items)
        X, y, info = OhlcvProcessor.get_ds_raw_data(stock_symbol, window_size=10)
        print('X: {0};'.format(X.shape))
        print('y: {0};'.format(y.shape))
        print('info: {0};'.format(len(info)))

    def test_t001(self):
        window_size = 3
        log_1d_datas = [
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [2.1, 2.2, 2.3, 2.4, 2.5],
            [3.1, 3.2, 3.3, 3.4, 3.5],
            [4.1, 4.2, 4.3, 4.4, 4.5],
            [5.1, 5.2, 5.3, 5.4, 5.5],
            [6.1, 6.2, 6.3, 6.4, 6.5],
            [7.1, 7.2, 7.3, 7.4, 7.5]
        ]
        ldd_size = len(log_1d_datas)
        print('ldd_size: {0};'.format(ldd_size))
        X_raw = []
        for pos in range(window_size, ldd_size, 1):
            item = []
            for idx in range(pos-window_size, pos):
                item += log_1d_datas[idx]
            item += log_1d_datas[pos]
            X_raw.append(item)
        print(X_raw)
