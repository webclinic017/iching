# 50ETF日行情数据集类
import sys
import numpy as np
import torch
import torch.utils.data.dataset as Dataset
#
from apps.sop.sop_config import SopConfig
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
        raw_dates = list(date_set)
        list.sort(raw_dates, reverse=False)
        list.sort(self.key_list, reverse=False)
        # 求出系统日历
        self.dates = []
        for idx in range(SopConfig.lookback_num - 1, len(raw_dates)):
            self.dates.append(raw_dates[idx])
        # 获取50ETF指数日行情数据
        index_ds = Sh50etfIndexDataSource()
        index_df = index_ds.get_daily_data(raw_dates[0], raw_dates[-1])
        raw_X = [] # 一天一行形式
        for idx in range(len(raw_dates)):
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
            try:
                index_rec = index_df.loc[raw_dates[idx]]
                date_row += [
                    float(index_rec['open']), float(index_rec['high']), 
                    float(index_rec['low']), float(index_rec['close']), 
                    float(index_rec['volume'])
                ]
            except KeyError as ke:
                print('Encounter KeyError: {0};'.format(ke))
                date_row += [0.0, 0.0, 0.0, 0.0, 0]
            raw_X.append(date_row)
        X_n = [] # 向前5天行情组成一行
        for idx in range(SopConfig.lookback_num -1, len(raw_X)):
            tick_data = []
            for j in range(SopConfig.lookback_num-1, -1, -1):
                tick_data += raw_X[idx - j]
            X_n.append(tick_data)
        X = np.array(X_n, dtype=np.float32)
        y = np.zeros((X.shape[0],))
        r = np.zeros((X.shape[0],))
        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(r)