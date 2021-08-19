#
import unittest
from apps.fmts.ds.ohlcv_processor import OhlcvProcessor

class TOhlcvProcessor(unittest.TestCase):
    def test_draw_close_price_curve_001(self):
        stock_symbol = 'sh600260'
        OhlcvProcessor.draw_close_price_curve(stock_symbol, mode=OhlcvProcessor.PCM_TICK)