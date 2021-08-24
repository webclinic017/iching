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

    @staticmethod
    def gen_1d_log_diff_norm(stock_symbol, items):
        '''
        从原始行情数据，求出一阶对数收益率log(day2)-log(day1)，然后求出每列均值和标准差，利用
        (x-mu)/std进行标准化，分别保存原始信息和归整后信息
        '''
        datas = np.array([x[1:] for x in items])
        log_ds = np.log(datas)
        log_diff = np.diff(log_ds, n=1, axis=0)
        log_diff_mu = np.mean(log_diff, axis=0)
        log_diff_std = np.std(log_diff, axis=0)
        ld_ds = (log_diff - log_diff_mu) / log_diff_std
        # 保存原始信息
        raw_file = './apps/fmts/data/{0}_1m_raw.txt'.format(stock_symbol)
        with open(raw_file, 'w', encoding='utf-8') as fd:
            for item in items[1:]:
                fd.write('{0},{1},{2},{3},{4},{5}\n'.format(item[0], item[1], item[2], item[3], item[4], item[5]))
        # 保存规整化后数据
        ld_file = './apps/fmts/data/{0}_1m_ld.csv'.format(stock_symbol)
        np.savetxt(ld_file, ld_ds)

    @staticmethod
    def get_ds_raw_data(stock_symbol, window_size=10):
        '''
        获取数据集所需数据
        stock_symbol 股票代码
        window_size 从当前时间点向前看多少个时间点
        返回值 
            X 连续11个时间点的OHLCV的数据，形状为n*55，一阶Log差分形式
            y 某个时间点及其前10个时间点行情数据组成的shapelet对应的行情（按Box方式确定）：0-震荡；1-上升；2-下跌；
            info 当前时间刻行情的真实值
        '''
        print('获取数据集数据')
        # 获取归整化行情数据
        log_1d_datas = []
        log_1d_file = './apps/fmts/data/{0}_1m_ld.csv'.format(stock_symbol)
        with open(log_1d_file, 'r', encoding='utf-8') as fd:
            for row in fd:
                row = row.strip()
                arrs = row.split(' ')
                item = [arrs[0], arrs[1], arrs[2], arrs[3], arrs[4]]
                log_1d_datas.append(item)
        X = np.array(log_1d_datas)
        print('X: {0};'.format(X.shape))
        # 获取日期和真实行情数值
        raw_datas = []
        raw_data_file = './apps/fmts/data/{0}_1m_raw.txt'.format(stock_symbol)
        with open(raw_data_file, 'r', encoding='utf-8') as fd:
            for row in fd:
                row = row.strip()
                arrs = row.split(' ')
                raw_datas.append(arrs[0])
        print('日期：{0};'.format(raw_datas))

                





    def _draw_date_price_curve(x: List, y: List) -> None:
        '''
        给制横轴为时间的价格变化折线图
        '''
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
        '''
        绘制横轴为行情数据序号的价格变化折线图
        '''
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
