#
from __future__ import print_function
import datetime as dt
import pandas as pd
#from sklearn.qda import QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from apps.ots.strategy.strategy_base import StrategyBase
from apps.ots.event.ots_event import OtsEvent
from apps.ots.event.signal_event import SignalEvent
from apps.ots.ds.hist_ashar_csv_ds import HistAsharCsvDs
from apps.ots.broker.simulated_order_executor import SimulatedOrderExecutor
from apps.ots.strategy.portfolio import Portfolio
from apps.ots.backtest import Backtest
from apps.ots.ds.ds_util import DsUtil

class SnpDailyForecastStrategy(StrategyBase):
    '''
    S&P 500日K线交易策略，使用Quadratic Discriminant Analyser算法
    '''
    def __init__(self, bars, events):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.model_start_date = dt.datetime(2001, 1, 10)
        self.model_end_date = dt.datetime(2005, 12, 31)
        self.model_test_start_date = dt.datetime(2005, 1, 1)
        self.long_market = False
        self.short_market = False
        self.bar_index = 0
        self.model = self.create_symbol_forecast_model()
        # ?????????????
        self.datetime_now = dt.datetime.utcnow() # ??????????????
        # ??????????????

    def create_symbol_forecast_model(self):
        snpret = DsUtil.create_lagged_series(
            self.symbol_list[0], self.model_start_date,
            self.model_end_date, lags=5
        )
        X = snpret[['lag1', 'lag2']] # 只用之前两天的数据
        y = snpret['Direction']
        start_test = self.model_test_start_date
        X_train = X[X.index<start_test]
        X_test = X[X.index>=start_test]
        y_train = y[y.index<start_test]
        y_test = y[y.index>=start_test]
        print('X_train: {0}; {1};'.format(type(X_train), X_train))
        print('y_train: {0}; {1};'.format(type(y_train), y_train))
        model = QuadraticDiscriminantAnalysis()
        model.fit(X_train, y_train)
        return model

    def calculate_signals(self, event):
        '''
        基于市场数据计算SignalEvent
        '''
        sym = self.symbol_list[0]
        curr_dt = self.datetime_now
        if event.type == OtsEvent.ET_MARKET:
            self.bar_index += 1
            if self.bar_index>5: # 等待5条数据才能取前5条
                lags = self.bars.get_latest_bar_values(
                    self.symbol_list[0], 'returns', N=3
                )
                pred_series = pd.Series({ # 第一阶是当前的
                    'log1': lags[1]*100.0,
                    'lag2': lags[2]*100.0
                })
                pred = self.model.predict(pred_series)
                if pred>0 and not self.long_market:
                    self.long_market = True
                    signal_event = SignalEvent(1, sym, dt, 'LONG', 1.0)
                    self.events.put(signal_event)
                if pred<0 and self.long_market:
                    self.long_market = False
                    signal_event = SignalEvent(1, sym, dt, 'EXIT', 1.0)
                    self.events.put(signal_event)
        