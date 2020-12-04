#
from apps.ots.broker.simulated_order_executor import SimulatedOrderExecutor
from apps.ots.ds.hist_ashar_csv_ds import HistAsharCsvDs
from apps.ots.strategy.mac_strategy import MacStrategy
from apps.ots.strategy.portfolio import Portfolio
from apps.ots.backtest import Backtest
import datetime as dt

class OtsApp(object):
    def __init__(self):
        self.refl = ''

    def startup(self, args={}):
        csv_dir = './data'
        symbol_list = ['AAPL']
        initial_capital = 100000.0
        heartbeat = 0
        start_date = dt.datetime(2019, 1, 1, 0, 0, 0)
        backtest = Backtest(csv_dir, symbol_list, initial_capital, 
                    heartbeat, start_date, HistAsharCsvDs, 
                    SimulatedOrderExecutor, Portfolio, MacStrategy)
        backtest.startup()