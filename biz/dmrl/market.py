#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import akshare as ak
#
from biz.dmrl.app_config import AppConfig

class Market(Dataset):
    DS_MODE_FULL = 0
    DS_MODE_TRAIN = 1
    DS_MODE_VAL = 2
    DS_MODE_TEST = 3

    def __init__(self, stock_symbol):
        self.name = 'apps.dmrl.maml.aks_ds.AksDs'
        self.X, self.y = self.load_ds_from_txt(stock_symbol=stock_symbol)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    
    def load_ds_from_txt(self, stock_symbol):
        '''
        从文本文件中读出行情数据集，所有股票以字典形式返回
        '''
        X_file = './data/aks_ds/{0}_X.txt'.format(stock_symbol)
        X = np.loadtxt(X_file, delimiter=',', encoding='utf-8')
        y_file = './data/aks_ds/{0}_y.txt'.format(stock_symbol)
        y = np.loadtxt(y_file, delimiter=',', encoding='utf-8')
        return X, y