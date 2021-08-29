#
import random
import numpy as np
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
        X, y, info = OhlcvProcessor.get_ds_raw_data(stock_symbol, window_size=10, forward_size=100)
        print('X: {0};'.format(X.shape))
        print('y: {0};'.format(y.shape))
        print('info: {0};'.format(len(info)))

    def test_t001(self):
        stock_symbol = 'sh600260'
        quotation = OhlcvProcessor.get_quotations(stock_symbol)
        print('v0.0.1 quotation: {0};'.format(quotation.shape))

    def test_get_market_state001(self):
        '''
        测试上涨行情识别
        '''
        print('test get market state method')
        y = np.zeros((10,), dtype=np.int64)
        quotation = np.array([
            [0.01, 0.02, 0.03, 3.1, 0.04], 
            [0.01, 0.02, 0.03, 3.16, 0.04], 
            [0.01, 0.02, 0.03, 3.16, 0.04], 
            [0.01, 0.02, 0.03, 3.17, 0.04], 
            [0.01, 0.02, 0.03, 3.18, 0.04], 
            [0.01, 0.02, 0.03, 3.185, 0.04], 
            [0.01, 0.02, 0.03, 3.19, 0.04], 
            [0.01, 0.02, 0.03, 3.22, 0.04], 
            [0.01, 0.02, 0.03, 3.21, 0.04], 
            [0.01, 0.02, 0.03, 3.2, 0.04], 
            [0.01, 0.02, 0.03, 3.19, 0.04], 
            [0.01, 0.02, 0.03, 3.18, 0.04], 
            [0.01, 0.02, 0.03, 3.181, 0.04], 
            [0.01, 0.02, 0.03, 3.186, 0.04], 
            [0.01, 0.02, 0.03, 3.179, 0.04], 
            [0.01, 0.02, 0.03, 3.183, 0.04]
        ])
        window_size = 3
        forward_size = 4
        OhlcvProcessor.get_market_state(y, quotation, window_size, forward_size)
        print('y: {0}; {1};'.format(y.shape, y))

    def test_get_market_state002(self):
        msg = input('please input msg:')
        print('msg: {0};'.format(msg))
        i_debug = 1
        if 1==i_debug:
            return
        random.seed(1.0)
        y = np.zeros((10,), dtype=np.int64)
        quotation_raw = []
        cnt_y = 10
        curr_price = 5.36
        high_delta = 1.01
        low_delta = 0.995
        window_size = 3
        forward_size = 4
        cnt = cnt_y+window_size+forward_size-1
        for idx in range(cnt):
            close_price = random.uniform(curr_price*low_delta, curr_price*high_delta)
            item = [0.01, 0.02, 0.03, close_price, 0.04]
            quotation_raw.append(item)
        print('quotation_raw: {0}: {1};'.format(len(quotation_raw), quotation_raw))
        quotation = np.array(quotation_raw)
        OhlcvProcessor.get_market_state(y, quotation, window_size, forward_size)
        print('y: {0}; {1};'.format(y.shape, y))
        x = range(cnt)
