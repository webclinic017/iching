#
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Covid2019Ds(Dataset):
    def __init__(self, csv_file, mode='train', target_only=False):
        '''
        读取数据集CSV文件，前40列为州编号的one hot编码，18列为大前天观测值，18列前天观测值，
        18列为今天观测值（测试集上仅有17列，第18列是需要预测的值）
        @param mode 可以为：train, valid, test
        @param target_only 只使用前两天的检测结果列去预测
        '''
        self.mode = mode
        with open(csv_file, 'r') as fd:
            data = list(csv.reader(fd))
            data = np.array(data[1:])[:, 1:].astype(float)
        # 准备特征的索引号序列
        X_dim = 93 # 93 = 40 + 18 + 18 + 17
        state_dim = 40
        tv_percent = 10 # 训练样本集与验证样本集比例
        if not target_only:
            feats = list(range(X_dim))
        else:
            feats = list(range(state_dim))
            feats.append(57) # 大前天结果（第18列）
            feats.append(75) # 前天结果（第18列）
        if 'test' == self.mode:
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            self.X = torch.FloatTensor(data[:, feats])
        else:
            data = data[:, feats]
            target = data[:, -1]
            if 'train' == mode:
                indices = [i for i in range(len(data)) if i%tv_percent != 0]
            elif 'valid' == mode:
                indices = [i for i in range(len(data)) if i%tv_percent == 0]
        self.X = torch.FloatTensor(data[indices])
        self.y = torch.FloatTensor(target[indices])
        # 特征归一化：(x-mu)/std
        self.X[:, 40:] = (self.X[:, 40:] - self.X[:, 40:].mean(dim=0, keepdim=True)) \
                    / self.X[:, 40:].std(dim=0, keepdim=True)
        self.dim = self.X.shape[1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.mode in ['train', 'test']:
            return self.X[index], self.y[index]
        else:
            return self.X[index]