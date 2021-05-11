#
import os
import numpy as np
import pandas as pd
import akshare as ak
from apps.ots.ds.base_ds import BaseDs
from apps.ots.event.market_event import MarketEvent

class HistAshareCsvHftDs(BaseDs):
    ''' 读取A股行情CSV文件，模拟现实情况 '''
    def __init__(self, events, csv_dir, symbol_list):
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True # False时退出回测过程
        self.bar_index = 0 # 当前数据条索引号
        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        ''' 打开CSV文件，将其转化为panda的DataFrame，我们以Yahoo数据源 '''
        comb_index = None
        for s in self.symbol_list:
            print('csv_dir: {0}; file: {1};'.format(self.csv_dir, s))
            self.symbol_data[s] = pd.io.parsers.read_csv(
                '{0}/{1}.csv'.format(self.csv_dir, s),
                header=0, index_col=1, parse_dates=True,
                names = ['day', 'open', 'high', 'low', 'close', 'volume']
            ).sort_values(by='day')
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)
            self.latest_symbol_data[s] = []
        print('comb_index: {0};'.format(comb_index))
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].reindex(index = comb_index, method='pad')
            self.symbol_data[s]['return'] = self.symbol_data[s]['close'] / self.symbol_data[s]['close'].shift(1) - 1
            self.symbol_data[s] = self.symbol_data[s].iterrows()

    def _get_new_bar(self, symbol):
        ''' 返回最新数据条 '''
        for b in self.symbol_data[symbol]:
            yield b

    def get_latest_bar(self, symbol):
        '''
        从最新的symbol_list中返回最新数据条目
        '''
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print('该期权在历史数据中不存在')
            raise
        else:
            return bars_list[-1]

    def get_latest_bars(self, symbol, N=1):
        ''' 从最新数据列表中获取N条数据，若不足时则返回N-k条 '''
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print('该期权在历史数据中不存在')
            raise
        else:
            return bars_list[-N:]

    def get_latest_bar_dt(self, symbol):
        ''' 返回最新数据条的对应时间 '''
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print('该期权在历史数据中不存在')
            raise
        else:
            return bars_list[-1][0]

    def get_latest_bar_value(self, symbol, val_type):
        ''' 返回最新数据条某一列的指定值：Open, High, Low, Close, Volume, OI '''
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            print('该期权在历史数据中不存在')
            raise
        else:
            return getattr(bars_list[-1][1], val_type)

    def get_latest_bar_values(self, symbol, val_type, N=1):
        ''' 返回历史数据条中N条数据中指定列的指定值 '''
        try:
            bars_list = self.get_latest_bars(symbol, N)
        except KeyError:
            print('该期权在历史数据中不存在')
            raise
        else:
            return np.array([getattr(b[1], val_type) for b in bars_list])

    def update_bars(self):
        ''' 将最近的数据条放入latest_symbol_data中 '''
        for s in self.symbol_list:
            try:
                bar = next(self._get_new_bar(s))
            except StopIteration:
                self.continue_backtest = False
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
        self.events.put(MarketEvent())

    @staticmethod
    def download_ashare_minute_data(symbol):
        stock_zh_a_minute_df = ak.stock_zh_a_minute(symbol=symbol, period='1', adjust="hfq")
        print('type: {0}; data: {1};'.format(type(stock_zh_a_minute_df), stock_zh_a_minute_df))
        stock_zh_a_minute_df.to_csv('./data/{0}.csv'.format(symbol))