# 用于强化学习回测环境数据集
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import akshare as ak
#
from biz.dmrl.app_config import AppConfig
from biz.dmrl.aks_util import AksUtil

class FmtsMarketDs(Dataset):
    def __init__(self, stock_symbol):
        self.name = 'apps.dmrl.maml.aks_ds.AksDs'
        self.X, self.y, self.trade_dates = AksUtil.generate_stock_ds(stock_symbol, ds_mode=AksUtil.DM_RL, draw_line=False)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.trade_dates[idx]

    def get_trade_date(self, current_step):
        '''
        获取当前交易点的日期字符串
        '''
        return self.trade_dates[current_step - 1]