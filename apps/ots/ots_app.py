#
import datetime as dt
from apps.ots.strategy.intraday_mr_strategy import IntradayMrStrategy
from apps.ots.strategy.snp_daily_forecast_strategy import SnpDailyForecastStrategy
from apps.ots.broker.simulated_order_executor import SimulatedOrderExecutor
from apps.ots.ds.hist_ashar_csv_ds import HistAsharCsvDs
from apps.ots.strategy.mac_strategy import MacStrategy
from apps.ots.strategy.portfolio import Portfolio
from apps.ots.backtest import Backtest
from apps.ots.ds.hist_ashare_csv_hft_ds import HistAshareCsvHftDs
from apps.ots.strategy.portfolio_hft import PortfolioHft

class OtsApp(object):
    def __init__(self):
        self.refl = ''

    def startup(self, args={}):
        #self.mac_main(args)
        #self.snp_daily_forcast_strategy_main(args)
        self.intraday_mr_strategy_main(args)
        #HistAshareCsvHftDs.download_ashare_minute_data('sh601939')

    def mac_main(self, args):
        csv_dir = './data'
        symbol_list = ['AAPL']
        initial_capital = 100000.0
        heartbeat = 0
        start_date = dt.datetime(2019, 1, 1, 0, 0, 0)
        backtest = Backtest(csv_dir, symbol_list, initial_capital, 
                    heartbeat, start_date, HistAsharCsvDs, 
                    SimulatedOrderExecutor, Portfolio, MacStrategy)
        backtest.startup()

    def snp_daily_forcast_strategy_main(self, args={}):
        csv_dir = './data'
        symbol_list = ['SPY']
        initial_capital = 100000.0
        heartbeat = 0
        start_date = dt.datetime(2006, 1, 1)
        backtest = Backtest(csv_dir, symbol_list, initial_capital,
                    heartbeat, start_date, HistAsharCsvDs,
                    SimulatedOrderExecutor, Portfolio, SnpDailyForecastStrategy
        )
        backtest.startup()

    def intraday_mr_strategy_main(self, args={}):
        print('日内分钟级数据交易对统计套利策略')
        csv_dir = './data'
        symbol_list = ['sh601939', 'sh600000']
        initial_capital = 100000.0
        heartbeat = 0.0
        start_date = dt.datetime(2019, 1, 1, 10, 41, 0)
        backtest = Backtest(
            csv_dir, symbol_list, initial_capital, heartbeat,
            start_date, HistAshareCsvHftDs, SimulatedOrderExecutor,
            PortfolioHft, IntradayMrStrategy
        )
        backtest.startup()