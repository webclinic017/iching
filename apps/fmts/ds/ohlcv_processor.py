# 处理OHLCV数据，供OhlcvDataset类使用
import datetime
from typing import List
from typing import Tuple
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
    def get_ds_raw_data(stock_symbol: str, window_size: int=10, forward_size: int=100) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        '''
        获取数据集所需数据
        stock_symbol 股票代码
        window_size 从当前时间点向前看多少个时间点
        forward_size 向后看多少个时间点确定市场行情是上涨、下跌和震荡
        返回值 
            X 连续11个时间点的OHLCV的数据，形状为n*55，一阶Log差分形式
            y 某个时间点及其前10个时间点行情数据组成的shapelet对应的行情（按Box方式确定）：0-震荡；1-上升；2-下跌；
            info 当前时间刻行情的真实值
        '''
        print('获取数据集数据')
        # 获取行情数据
        quotations = OhlcvProcessor.get_quotations(stock_symbol)
        # 获取归整化行情数据
        log_1d_datas = []
        log_1d_file = './apps/fmts/data/{0}_1m_ld.csv'.format(stock_symbol)
        with open(log_1d_file, 'r', encoding='utf-8') as fd:
            for row in fd:
                row = row.strip()
                arrs = row.split(' ')
                item = [arrs[0], arrs[1], arrs[2], arrs[3], arrs[4]]
                log_1d_datas.append(item)
        # 
        ldd_size = len(log_1d_datas) - forward_size
        X_raw = []
        for pos in range(window_size, ldd_size, 1):
            item = []
            for idx in range(pos-window_size, pos):
                item += log_1d_datas[idx]
            item += log_1d_datas[pos]
            X_raw.append(item)
        X = np.array(X_raw, dtype=np.float32)
        ds_X_csv = './apps/fmts/data/{0}_1m_X.csv'.format(stock_symbol)
        np.savetxt(ds_X_csv, X, delimiter=',')
        # 获取行情状态
        y = np.zeros((X.shape[0],), dtype=np.int64)
        #OhlcvProcessor.get_market_state(y, quotations, window_size, forward_size)
        # 获取日期和真实行情数值
        raw_datas = []
        raw_data_file = './apps/fmts/data/{0}_1m_raw.txt'.format(stock_symbol)
        seq = 0
        with open(raw_data_file, 'r', encoding='utf-8') as fd:
            for row in fd:
                if seq >= window_size and seq<ldd_size:
                    row = row.strip()
                    arrs = row.split(',')
                    item = [arrs[0], arrs[1], arrs[2], arrs[3], arrs[4], arrs[5]]
                    raw_datas.append(item)
                seq += 1
        a1 = len(raw_datas)
        return X[:a1], y[:a1], raw_datas

    @staticmethod
    def get_quotations(stock_symbol: str) -> np.ndarray:
        '''
        获取原始行情数据，格式：[..., [open, high, low, close, volume], ...]，并以numpy数组形式返回
        '''
        raw_data = AkshareDataSource.get_minute_bars(stock_symbol)
        q_data = [x[1:] for x in raw_data]
        return np.array(q_data, dtype=np.float32)

    @staticmethod
    def get_market_state(y: np.ndarray, quotation: np.ndarray, window_size: int, forward_size: int) -> None:
        '''
        针对收盘价，从window_size处开始，向后看forward_size条记录，上限为当前收盘价*1.01，下限为当前收盘价*0.95，当
        在forward_size窗口内收盘价高于上限时返回0表示上升行情需要买入，低于下限时返回1表示下跌行情需要卖出，
        否则返回2表示震荡行情，将该值写入y中
        '''
        cnt = y.shape[0]
        q_cnt = quotation.shape[0]
        for idx in range(0, cnt, 1):
            curr_price = quotation[idx+window_size][3]
            high_limit = curr_price * 1.01
            low_limit = curr_price * 0.995
            market_regime = 2
            for pos in range(idx+window_size+1, idx+window_size+1+forward_size, 1):
                future_price = quotation[pos][3]
                if future_price >= high_limit:
                    market_regime = 0
                elif future_price <= low_limit:
                    market_regime = 1
            y[idx] = market_regime
                





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
