#
from __future__ import print_function
import datetime as dt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import product
#
from apps.ots.strategy.strategy_base import StrategyBase
from apps.ots.event.ots_event import OtsEvent
from apps.ots.event.signal_event import SignalEvent
from apps.ots.backtest import Backtest
from apps.ots.ds.hist_ashare_csv_hft_ds import HistAshareCsvHftDs
from apps.ots.strategy.portfolio_hft import PortfolioHft
from apps.ots.broker.simulated_order_executor import SimulatedOrderExecutor

class IntradayMrStrategy(StrategyBase):
    '''
    使用最小二乘法进行Rolling的线性回归来确定两股票的对冲比例，接着计算时间序列列差的zscore，
    分析是否落在上、下限之间，然后进行配对交易，生成交易或退出信号
    '''

    def __init__(self, bars, events, ols_window=60, zscore_low=2.5, zscore_high=5.0):
        '''
        取100分钟（因为是分钟数据）为查看周期，当小于zscore_low或大于zscore_high时
        进入交易，否则退出交易
        '''
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.ols_window = ols_window
        self.zscore_low = zscore_low
        self.zscore_high = zscore_high
        # 配对股票：建设银行、浦发银行
        self.pair = ['sh601939', 'sh600000']
        self.curr_dt = dt.datetime.utcnow()
        self.long_market = False
        self.short_market = False

    def calculate_xy_signals(self, zscores):
        '''
        计算交易对的信号配对，传递给信号生成方法
        '''
        y_signal = None
        x_signal = None
        p0 = self.pair[0]
        p1 = self.pair[1]
        curr_dt = self.curr_dt
        hr = abs(self.hedge_ratio)
        zscore_latest = zscores[-1]
        bar_date = self.bars.get_latest_bar_dt(self.symbol_list[0])
        if zscore_latest <= -self.zscore_high and self.long_market:
            self.long_market = True
            y_signal = SignalEvent(1, p0, curr_dt, 'LONG', 1.0)
            x_signal = SignalEvent(1, p1, curr_dt, 'SHORT', hr)
            print('{0}：买入建设银行+卖出浦发银发'.format(bar_date))
        elif abs(zscore_latest)<=self.zscore_low and self.long_market:
            self.long_market = False
            y_signal = SignalEvent(1, p0, curr_dt, 'EXIT', 1.0)
            x_signal = SignalEvent(1, p1, dt, 'EXIT', 1.0)
            print('{0}：卖出建设银行+卖出浦发银行'.format(bar_date))
        elif zscore_latest>=self.zscore_high and not self.short_market:
            self.short_market = True
            y_signal = SignalEvent(1, p0, curr_dt, 'SHORT', 1.0)
            x_signal = SignalEvent(1, p1, curr_dt, 'LONG', hr)
            print('{0}：卖出建设银行+买入浦发银行'.format(bar_date))
        elif abs(zscore_latest)<=self.zscore_low and self.short_market:
            self.short_market = False
            y_signal = SignalEvent(1, p0, curr_dt, 'EXIT', 1.0)
            x_signal = SignalEvent(1, p1, curr_dt, 'EXIT', 1.0)
            print('{0}：卖出建设银行+卖出浦发银行'.format(bar_date))
        return y_signal, x_signal

    def calculate_signals_for_pairs(self):
        '''
        基于均值回归策略来生成一组信号，同时使用ols计算交易对的对冲比例
        '''
        y = self.bars.get_latest_bar_values(self.pair[0], 'close', self.ols_window)
        x = self.bars.get_latest_bar_values(self.pair[1], 'close', self.ols_window)
        if y is not None and x is not None:
            if (len(y)>=self.ols_window and len(x)>=self.ols_window):
                self.hedge_ratio = sm.OLS(y, x).fit().params[0]
                spread = y-self.hedge_ratio*x
                zscore_latest = ((spread - spread.mean()) / spread.std())
                y_signal, x_signal = self.calculate_xy_signals(zscore_latest)
                if y_signal is not None and x_signal is not None:
                    self.events.put(y_signal)
                    self.events.put(x_signal)

    def calculate_signals(self, event):
        '''
        基于市场数据，计算SignalEvent
        '''
        if event.type == OtsEvent.ET_MARKET:
            self.calculate_signals_for_pairs()