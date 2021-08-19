# 处理OHLCV数据，供OhlcvDataset类使用
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from apps.fmts.ds.akshare_data_source import AkshareDataSource


class OhlcvProcessor(object):
    def __init__(self):
        self.name = 'apps.fmts.ds.ohlcv_processor.OhlcvProcessor'

    @staticmethod
    def draw_close_price_curve(stock_symbol: str) -> None:
        '''
        绘制收盘价折线图
        '''
        data = AkshareDataSource.get_minute_bars(stock_symbol=stock_symbol)
        x_raw = [v[0] for v in data[0:250]]
        x = [datetime.datetime.strptime(di, '%Y-%m-%d %H:%M:%S') for di in x_raw]
        y = [v[4] for v in data[0:250]]
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        figmanager = plt.get_current_fig_manager()
        figmanager.window.state('zoomed')    #最大化
        #axes.plot(x, y, linestyle='-', color='#DE6B58', marker='x', linewidth=1.5)
        axes.plot_date(x, np.array(y), '-', label='Net Worth')
        axes.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
        #axes.fmt_xdata =  
        #fig.autofmt_xdate()
        plt.gcf().autofmt_xdate()
        plt.show()