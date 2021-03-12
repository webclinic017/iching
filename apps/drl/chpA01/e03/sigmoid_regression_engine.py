#
import torch
from torch.utils.data import DataLoader
from apps.drl.chpA01.e03.sigmoid_regression_ds import SigmoidRegressionDs

class SigmoidRegressionEngine(object):
    def __init__(self):
        self.name = ''

    def train(self):
        pass

    def exp(self):
        ds = SigmoidRegressionDs()
        total = len(ds)
        train_count = int(total * 0.7)
        valid_count = int(total * 0.2)
        test_count = total - train_count - valid_count
        train_ds, valid_ds, test_ds = torch.utils.data.random_split(
            ds, (train_count, valid_count, test_count)
        )
        train_batch_size = 16
        num_workers = 4
        train_dl = DataLoader(train_ds, train_batch_size, 
                shuffle=True,
                drop_last=False, num_workers=num_workers, pin_memory=True)
        valid_batch_size = 32
        valid_dl = DataLoader(valid_ds, valid_batch_size, 
                shuffle=False,
                drop_last=False, num_workers=num_workers, pin_memory=True)
        test_batch_size = 32
        test_dl = DataLoader(test_ds, test_batch_size, 
                shuffle=False,
                drop_last=False, num_workers=num_workers, pin_memory=True)
        for X, y_hat in train_dl:
            print('X:{0}; y:{1};'.format(X, y_hat))