# 处理日、小时、分钟、秒级Open，High, Low, Close, Volume数据集，这里
# 仅处理复权后数据，所有数值均采用一阶log相减形式（对数收益率），
# 然后进行归一化，(x-mu) / std ，数据格式为：
#   X: Open'1, High'1, Low'1, Close'1, Volume'1, Open'2, High'2, Low'2, Close'2, Volume'2, ..., Open'w, High'w, Low'w, Close'w, Volume'w，
#       其中第w天为当前日期，向前推w天
#   y: number 所处市场行情
#   info：日期时间
from numpy.core import overrides
from torch.utils.data import Dataset

class OhlcvDataset(Dataset):
    def __init__(self, X, y, info):
        self.name = 'apps.fmts.ds.ohlcv_ds.OhlcvDs'
        self.X = X
        self.y = y
        self.info = info

    @overrides
    def __len__(self):
        return self.X.shape[0]

    @overrides
    def __item__(self, idx):
        return self.X[idx], self.y[idx], self.info[idx]