#
import iqt.env.default as default
from iqt.feed.core import Stream, DataFeed
from iqt.data.cdd import CryptoDataDownload
from iqt.oms.wallets import Portfolio, Wallet
from iqt.oms.exchanges import Exchange
from iqt.oms.services.execution.simulated import execute_order
from iqt.oms.instruments import USD, BTC, ETH, LTC

class DctDs(object):
    def __init__(self):
        self.name = 'biz.dct.dct_ds.DctDs'

    def load_data(self):
        print('载入数据...')
        cdd = CryptoDataDownload()
        # 获取bitfinex行情数据
        bitfinex_btc = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
        print('step 1')
        bitfinex_eth = cdd.fetch("Bitfinex", "USD", "ETH", "1h")
        print('step 2')
        # 获取bitstamp行情数据
        bitstamp_btc = cdd.fetch("Bitstamp", "USD", "BTC", "1h")
        print('step 3')
        bitstamp_eth = cdd.fetch("Bitstamp", "USD", "ETH", "1h")
        print('step 4')
        bitstamp_ltc = cdd.fetch("Bitstamp", "USD", "LTC", "1h")
        print('step 5')
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
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print('    action={0}; {1};'.format(action, obs))

        df = portfolio.ledger.as_frame().head(7)
        print('result: {0};'.format(df))
        print('^_^ v0.0.4')