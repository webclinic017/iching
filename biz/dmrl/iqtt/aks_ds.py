#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import akshare as ak

class AksDs(Dataset):
    DS_MODE_FULL = 0
    DS_MODE_TRAIN = 1
    DS_MODE_VAL = 2
    DS_MODE_TEST = 3

    def __init__(self, stock_symbol, ds_mode=0, train_rate=0.0, val_rate=0.0, test_rate=0.0):
        self.name = 'apps.dmrl.maml.aks_ds.AksDs'
        X_raw, y_raw = self.load_ds_from_txt(stock_symbol=stock_symbol)
        if ds_mode == AksDs.DS_MODE_FULL: # 全部数据集
            self.X, self.y = X_raw, y_raw
        elif ds_mode == AksDs.DS_MODE_TRAIN: # 训练数据集
            start_pos = int(X_raw.shape[0] * train_rate)
            self.X = X_raw[:start_pos]
            self.y = y_raw[:start_pos]
        elif AksDs.DS_MODE_VAL == ds_mode: # 验证数据集
            start_pos = int(X_raw.shape[0] * train_rate)
            end_pos = start_pos + int(X_raw.shape[0]*val_rate)
            self.X = X_raw[start_pos : end_pos]
            self.y = y_raw[start_pos : end_pos]
        elif AksDs.DS_MODE_TEST == ds_mode: # 测试数据集
            start_pos = int(X_raw.shape[0] * train_rate) + int(X_raw.shape[0]*val_rate)
            self.X = X_raw[start_pos :]
            self.y = y_raw[start_pos :]

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