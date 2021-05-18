#
import iqt.env.default as default
from iqt.feed.core import Stream, DataFeed
from iqt.oms.wallets import Portfolio, Wallet
from iqt.oms.exchanges import Exchange
from iqt.oms.services.execution.simulated import execute_order
from iqt.oms.instruments import USD, BTC, ETH, LTC
from biz.dct.dct_ds import DctDs
# 
from biz.dct.iqc.iqt_simple_trader import IqtSimpleTrader

class DctApp(object):
    def __init__(self):
        self.name = 'apps.dct.dct_app.DctApp'

    def startup(self, args={}):
        print('数字货币交易系统 v0.0.1')
        #self.test_ledger()
        trader = IqtSimpleTrader()
        trader.startup()

    def test_ledger(self):
        print('test ledger')
        ds = DctDs()
        bitfinex_btc, bitfinex_eth, bitstamp_btc, bitstamp_eth, bitstamp_ltc = ds.load_data()
        # 定义交易所
        bitfinex = Exchange("bitfinex", service=execute_order)(
            Stream.source(list(bitfinex_btc['close'][-100:]), dtype="float").rename("USD-BTC"),
            Stream.source(list(bitfinex_eth['close'][-100:]), dtype="float").rename("USD-ETH")
        )
        bitstamp = Exchange("bitstamp", service=execute_order)(
            Stream.source(list(bitstamp_btc['close'][-100:]), dtype="float").rename("USD-BTC"),
            Stream.source(list(bitstamp_eth['close'][-100:]), dtype="float").rename("USD-ETH"),
            Stream.source(list(bitstamp_ltc['close'][-100:]), dtype="float").rename("USD-LTC")
        )
        # 定义在各个交易所的投资组合
        portfolio = Portfolio(USD, [
            Wallet(bitfinex, 1000000 * USD),
            Wallet(bitfinex, 1000 * BTC),
            Wallet(bitfinex, 500 * ETH),
            Wallet(bitstamp, 100000 * USD),
            Wallet(bitstamp, 500 * BTC),
            Wallet(bitstamp, 2000 * ETH),
            Wallet(bitstamp, 300 * LTC)
        ])
        # 定义数据源
        feed = DataFeed([
            Stream.source(list(bitstamp_eth['volume'][-100:]), dtype="float").rename("volume:/USD-ETH"),
            Stream.source(list(bitstamp_ltc['volume'][-100:]), dtype="float").rename("volume:/USD-LTC")
        ])
        # 生成环境
        env = default.create(
            portfolio=portfolio,
            action_scheme=default.actions.SimpleOrders(min_order_pct=0.0001),
            reward_scheme=default.rewards.SimpleProfit(),
            feed=feed
        )
        done = False
        obs = env.reset()
        env.action_scheme.min_order_pct = 0.0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

        df = portfolio.ledger.as_frame().head(7)
        print('result: {0};'.format(df))