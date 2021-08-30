#
import unittest
from torch.utils.data import DataLoader
from apps.fmts.ds.ohlcv_dataset import OhlcvDataset
from apps.fmts.ds.ohlcv_processor import OhlcvProcessor

class TOhlcvDataset(unittest.TestCase):
    def test_get_item(self):
        stock_symbol = 'sh600260'
        window_size = 10
        forward_size = 100
        X, y, info = OhlcvProcessor.get_ds_raw_data(stock_symbol, window_size, forward_size)
        ohlcv_ds = OhlcvDataset(X, y, info)
        print('样本数：{0};'.format(len(ohlcv_ds)))
        dl = DataLoader(ohlcv_ds, batch_size=1, shuffle=True, num_workers=0)
        X, y, info = next(iter(dl))
        quotation = OhlcvDataset.get_quotation_from_info(info)
        print('X: {0};'.format(X))
        print('y: {0};'.format(y))
        print('info: {0}; {1};'.format(quotation['date'], quotation['close']))
