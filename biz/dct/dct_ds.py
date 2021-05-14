#
from iqt.data.cdd import CryptoDataDownload

class DctDs(object):
    def __init__(self):
        self.name = 'biz.dct.dct_ds.DctDs'

    def load_data(self):
        print('载入数据...')
        cdd = CryptoDataDownload()
        # 获取bitfinex行情数据
        bitfinex_btc = cdd.fetch("Bitfinex", "USD", "BTC", "1h")
        print('载入bitfinex_btc...')
        bitfinex_eth = cdd.fetch("Bitfinex", "USD", "ETH", "1h")
        print('载入bitfinex_eth...')
        # 获取bitstamp行情数据
        bitstamp_btc = cdd.fetch("Bitstamp", "USD", "BTC", "1h")
        print('载入bitstamp_btc...')
        bitstamp_eth = cdd.fetch("Bitstamp", "USD", "ETH", "1h")
        print('载入bitstamp_eth...')
        bitstamp_ltc = cdd.fetch("Bitstamp", "USD", "LTC", "1h")
        print('载入bitstamp_ltc...')
        return bitfinex_btc, bitfinex_eth, bitstamp_btc, bitstamp_eth, bitstamp_ltc