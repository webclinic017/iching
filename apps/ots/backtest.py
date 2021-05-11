#
from __future__ import print_function
import time
import datetime as dt
import pprint
try:
    import Queue as queue
except ImportError:
    import queue
from apps.ots.event.ots_event import OtsEvent
from apps.ots.ots_util import OtsUtil

class Backtest(object):
    '''
    封装了事件驱动回测过程的实现
    '''
    def __init__(self, csv_dir, symbol_list, 
                initial_capital, heartbeat, 
                start_date, data_source, 
                order_executor, portfolio, strategy):
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date
        self.data_source_cls = data_source
        self.order_executor_cls = order_executor
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        self.events = queue.Queue()
        self.signalｓ = 0
        self.orderｓ = 0
        self.fills = 0
        self.num_strats = 1 # 策略为1
        self._generate_trading_instances()

    def _generate_trading_instances(self):
        '''
        从不同的类型中生成交易实例对象
        '''
        self.data_source = self.data_source_cls(self.events, self.csv_dir, self.symbol_list)
        self.strategy = self.strategy_cls(self.data_source, self.events)
        self.portfolio = self.portfolio_cls(self.data_source, self.events, self.start_date, self.initial_capital)
        self.order_executor = self.order_executor_cls(self.events)

    def _learn(self):
        print('^_^')

    def _run_backtest(self):
        '''
        执行回测
        '''
        self._learn()
        exit(1)
        i = 0
        while True:
            i += 1
            if self.data_source.continue_backtest:
                self.data_source.update_bars()
            else:
                break
            OtsUtil.step = i
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    # 若事件队列中无事件，则跳出内层循环
                    break
                else:
                    if event is None:
                        continue
                    if event.type == OtsEvent.ET_MARKET:
                        self.strategy.calculate_signals(event)
                        self.portfolio.update_timeindex(event) # 更新持仓
                    elif event.type == OtsEvent.ET_SIGNAL:
                        self.signalｓ += 1
                        self.portfolio.update_signal(event)
                    elif event.type == OtsEvent.ET_ORDER:
                        self.orders += 1
                        self.order_executor.execute_order(event)
                    elif event.type == OtsEvent.ET_FILL:
                        self.fills += 1
                        self.portfolio.update_fill(event)
            time.sleep(self.heartbeat)

    def _output_performance(self):
        '''
        输出回测业绩
        '''
        self.portfolio.create_equity_curve_dataframe()
        stats = self.portfolio.output_summary_stats()
        print('Equity Curve')
        print(self.portfolio.equity_curve.tail(10))
        pprint.pprint(stats)

    def startup(self):
        self._run_backtest()
        self._output_performance()
            