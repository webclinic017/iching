# Moving Average Crossing Strategy
from __future__ import print_function
import datetime as dt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from apps.ots.strategy.strategy_base import StrategyBase
from apps.ots.event.ots_event import OtsEvent
from apps.ots.event.signal_event import SignalEvent
from apps.ots.backtest import Backtest
from apps.ots.ds.hist_ashar_csv_ds import HistAsharCsvDs
from apps.ots.broker.simulated_order_executor import SimulatedOrderExecutor
from apps.ots.strategy.portfolio import Portfolio

class MacStrategy(StrategyBase):
    '''
    移动平均跨越策略实现，包括短期和长期移动平均值，默认短期、长期窗口为100天和400天
    '''
    def __init__(self, bars, events, short_window=100, long_window=400):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.short_window = short_window
        self.long_window = long_window
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        '''
        给bought字典增加键，对于所有代码设置为OUT
        '''
        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'
        return bought

    def calculate_signals(self, event):
        '''
        生成一组信号，进入市场信号：短期移动平均超过长期移动平均进场购买
        '''
        if event.type == OtsEvent.ET_MARKET:
            for s in self.symbol_list:
                bars = self.bars.get_latest_bar_values(s, 'adj_close', self.long_window)
                bar_date = self.bars.get_latest_bar_dt(s)
                if bars is not None and len(bars)>0:
                    if len(bars) < self.long_window:
                        return
                    short_sma = np.mean(bars[-self.short_window])
                    long_sma = np.mean(bars[-self.long_window])
                    curr_symbol = s
                    curr_dt = dt.datetime.utcnow()
                    signal_direction = ''
                    if short_sma > long_sma and self.bought[s] == 'OUT':
                        print('LONG（买入）：{0}'.format(bar_date))
                        signal_direction = 'LONG'
                        signal_event = SignalEvent(1, curr_symbol, curr_dt, signal_direction, 1.0)
                        self.events.put(signal_event)
                        self.bought[s] = 'LONG'
                    elif short_sma < long_sma and self.bought[s] == 'LONG':
                        print('SHORT（卖出）：{0}'.format(bar_date))
                        signal_direction = 'EXIT'
                        signal_event = SignalEvent(1, curr_symbol, curr_dt, signal_direction, 1.0)
                        self.events.put(signal_event)
                        self.bought[s] = 'OUT'