#
import iqt.env.default as default
from iqt.feed.core import Stream, DataFeed
from iqt.oms.wallets import Portfolio, Wallet
from iqt.oms.exchanges import Exchange
from iqt.oms.services.execution.simulated import execute_order
from iqt.oms.instruments import USD, BTC, ETH, LTC
from biz.dct.dct_ds import DctDs

class DctApp(object):
    def __init__(self):
        self.name = 'apps.dct.dct_app.DctApp'

    def startup(self, args={}):
        print('数字货币交易系统 v0.0.1')
        self.test_ledger()

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

        portfolio = Portfolio(USD, [
            Wallet(bitfinex, 10000 * USD),
            Wallet(bitfinex, 10 * BTC),
            Wallet(bitfinex, 5 * ETH),
            Wallet(bitstamp, 1000 * USD),
            Wallet(bitstamp, 5 * BTC),
            Wallet(bitstamp, 20 * ETH),
            Wallet(bitstamp, 3 * LTC)
        ])

        feed = DataFeed([
            Stream.source(list(bitstamp_eth['volume'][-100:]), dtype="float").rename("volume:/USD-ETH"),
            Stream.source(list(bitstamp_ltc['volume'][-100:]), dtype="float").rename("volume:/USD-LTC")
        ])

        env = default.create(
            portfolio=portfolio,
            action_scheme=default.actions.SimpleOrders(),
            reward_scheme=default.rewards.SimpleProfit(),
            feed=feed
        )
        done = False
        obs = env.reset()
        exit(1)


        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print('    action={0}; {1};'.format(action, obs))

        df = portfolio.ledger.as_frame().head(7)
        print('result: {0};'.format(df))