#
from iqt.data.cdd import CryptoDataDownload
from iqt.oms.exchanges.exchange import Exchange
from iqt.oms.services.execution.simulated import execute_order
from iqt.feed.core.base import Stream

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
        print('^_^')