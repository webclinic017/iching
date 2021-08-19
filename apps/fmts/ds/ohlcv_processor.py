# 处理OHLCV数据，供OhlcvDataset类使用
import datetime
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from apps.fmts.ds.akshare_data_source import AkshareDataSource


class OhlcvProcessor(object):
    # 价格折线图模式
    PCM_DATETIME = 1
    PCM_TICK = 2

    def __init__(self):
        self.name = 'apps.fmts.ds.ohlcv_processor.OhlcvProcessor'

    @staticmethod
    def draw_close_price_curve(stock_symbol: str, mode=1) -> None:
        '''
        绘制收盘价折线图，横轴为时间，纵轴为收盘价
        '''
        data = AkshareDataSource.get_minute_bars(stock_symbol=stock_symbol)
        x = [v[0] for v in data[0:1000]]
        y = [v[4] for v in data[0:1000]]
        if mode == OhlcvProcessor.PCM_DATETIME:
            OhlcvProcessor._draw_date_price_curve(x, y)
        else:
            OhlcvProcessor._draw_tick_price_curve(y)

    def _draw_date_price_curve(x: List, y: List) -> None:
        x = [datetime.datetime.strptime(di, '%Y-%m-%d %H:%M:%S') for di in x]
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
        # 最大化绘图窗口
        figmanager = plt.get_current_fig_manager()
        figmanager.window.state('zoomed')    #最大化
        # 绘制收盘价格折线图
        axes.plot_date(x, np.array(y), '-', label='Net Worth')
        # 设置横轴时间显示格式
        axes.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
        plt.gcf().autofmt_xdate()
        # 显示图像
        plt.show()
    
    def _draw_tick_price_curve(y: List) -> None:
        x = range(len(y))
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
        # 最大化绘图窗口
        figmanager = plt.get_current_fig_manager()
        figmanager.window.state('zoomed')    #最大化
        # 绘制收盘价格折线图
        plt.title('收盘价折线图')
        axes.set_xlabel('时间刻度')
        axes.set_ylabel('收盘价')
        axes.plot(x, np.array(y), '-', label='Net Worth')
        plt.show()
