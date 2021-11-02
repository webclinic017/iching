#
import random
import numpy as np
import matplotlib.pyplot as plt
import unittest
from apps.fmts.ds.akshare_data_source import AkshareDataSource
from apps.fmts.ds.ohlcv_processor import OhlcvProcessor

class TOhlcvProcessor(unittest.TestCase):
    def test_draw_close_price_curve_001(self):
        '''
        测试绘制以时间刻度为横轴的收盘价曲线，运行方式：
        python -m unittest apps.fmts.utcs.apps.fmts.ds.t_ohlcv_processor.TOhlcvProcessor.test_draw_close_price_curve_001 -v
        '''
        stock_symbol = AkshareDataSource.SS_BLYY
        OhlcvProcessor.draw_close_price_curve(stock_symbol, mode=OhlcvProcessor.PCM_TICK)
        
    def test_draw_close_price_curve_002(self):
        '''
        测试绘制以时间为横轴的收盘价曲线，运行方式：
        python -m unittest apps.fmts.utcs.apps.fmts.ds.t_ohlcv_processor.TOhlcvProcessor.test_draw_close_price_curve_002 -v
        '''
        stock_symbol = 'sh600260'
        OhlcvProcessor.draw_close_price_curve(stock_symbol, mode=OhlcvProcessor.PCM_DATETIME)

    def test_gen_1d_log_diff_norm_001(self):
        '''
        测试求规整后的一阶差分行情数据，运行方式：
        python -m unittest apps.fmts.utcs.apps.fmts.ds.t_ohlcv_processor.TOhlcvProcessor.test_gen_1d_log_diff_norm_001 -v
        '''
        stock_symbol = AkshareDataSource.SS_BLYY
        items = AkshareDataSource.get_minute_bars(stock_symbol=stock_symbol)
        OhlcvProcessor.gen_1d_log_diff_norm(stock_symbol, items)

    def test_get_ds_raw_data(self):
        '''
        获取用于模型训练的数据集
        python -m unittest apps.fmts.utcs.apps.fmts.ds.t_ohlcv_processor.TOhlcvProcessor.test_get_ds_raw_data -v
        '''
        stock_symbol = AkshareDataSource.SS_BLYY
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
        # 开始下标：window_size，结束下标：cnt-forward_size-1（包含）
        random.seed(1.0)
        y = np.zeros((10,), dtype=np.int64)
        quotation_raw = []
        cnt_y = 10
        curr_price = 5.36
        high_delta = 1.01
        low_delta = 0.995
        window_size = 3
        forward_size = 4
        cnt = cnt_y+window_size+forward_size
        for idx in range(cnt):
            close_price = random.uniform(curr_price*low_delta, curr_price*high_delta)
            item = [0.01, 0.02, 0.03, close_price, 0.04]
            quotation_raw.append(item)
        print('quotation_raw: {0}: {1};'.format(len(quotation_raw), quotation_raw))
        quotation = np.array(quotation_raw)
        OhlcvProcessor.get_market_state(y, quotation, window_size, forward_size)
        print('y: {0}; {1};'.format(y.shape, y))
        plt.ion()
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        for idx in range(cnt_y):
            plt.draw()
            plt.pause(0.1)
            plt.cla()
            # 绘制价格变化曲线
            x = range(cnt)
            close_prices = [ix[3] for ix in quotation]
            plt.plot(x, close_prices, color='goldenrod', marker='*')
            # 绘制最左侧竖线
            low_limit = quotation[idx+window_size][3]*low_delta
            high_limit = quotation[idx+window_size][3]*high_delta
            x1 = np.array([idx+window_size, idx+window_size])
            y1 = np.array([low_limit, high_limit])
            plt.plot(x1, y1, color='darkblue', marker='o')
            # 绘制上限
            x2 = np.array([idx+window_size, idx+window_size+forward_size])
            y2 = np.array([high_limit, high_limit])
            plt.plot(x2, y2, color='darkblue', marker='o')
            # 绘制下限
            x3 = np.array([idx+window_size, idx+window_size+forward_size])
            y3 = np.array([low_limit, low_limit])
            plt.plot(x3, y3, color='darkblue', marker='o')
            # 绘制右侧竖线
            x4 = np.array([idx+window_size+forward_size, idx+window_size+forward_size])
            y4 = np.array([low_limit, high_limit])
            plt.plot(x4, y4, color='darkblue', marker='o')
            # 标注市场状态
            plt.title('市场状态：{0};'.format(y[idx]))
            msg = input('please input msg:')
        plt.show(block=True)
