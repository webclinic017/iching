# 50ETF日行情数据集类
import sys
import numpy as np
import torch
import torch.utils.data.dataset as Dataset
#
from apps.sop.sh50etf_option_data_source import Sh50etfOptionDataSource
from apps.sop.ds.sh50etf_index_data_source import Sh50etfIndexDataSource

class Sh50etfDataset(Dataset.Dataset):
    def __init__(self):
        self.X, self.y, self.r = self._load_dataset()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.r[index]

    def _load_dataset(self):
        d_50etf = Sh50etfOptionDataSource()
        option_dict = d_50etf.get_data()
        # 获取日期列表
        date_set = set()
        self.key_list = []
        for key in option_dict.keys():
            self.key_list.append(key)
            for oc in option_dict[key]:
                date_set.add(oc[0])
        self.dates = list(date_set)
        list.sort(self.dates, reverse=False)
        list.sort(self.key_list, reverse=False)
        # 获取50ETF指数日行情数据
        index_ds = Sh50etfIndexDataSource()
        index_df = index_ds.get_daily_data(self.dates[0], self.dates[-1])
        raw_X = []
        for idx in range(len(self.dates)):
            date_row = []
            for key in self.key_list:
                oc = option_dict[key]
                if len(oc) > idx:
                    date_row += [float(oc[idx][1]), float(oc[idx][2]), 
                                float(oc[idx][3]), float(oc[idx][4]), 
                                float(oc[idx][5]), float(oc[idx][5]),
                                float(oc[idx][6]), float(oc[idx][7])]
                else:
                    date_row += [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            index_rec = index_df.loc[self.dates[idx]]
            date_row += [
                float(index_rec['open']), float(index_rec['high']), 
                float(index_rec['low']), float(index_rec['close']), 
                float(index_rec['volume'])
            ]
            raw_X.append(date_row)
        X = np.array(raw_X, dtype=np.float32)
        y = np.zeros((len(self.dates),))
        r = np.zeros((len(self.dates),))
        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(r)