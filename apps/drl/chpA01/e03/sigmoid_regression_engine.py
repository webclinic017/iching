#
from torch.utils.data import DataLoader
from apps.drl.chpA01.e03.sigmoid_regression_ds import SigmoidRegressionDs

class SigmoidRegressionEngine(object):
    def __init__(self):
        self.name = ''

    def train(self):
        pass

    def exp(self):
        ds = SigmoidRegressionDs()
        batch_size = 16
        mode = 'train'
        num_workers = 4
        dl = DataLoader(ds, batch_size, 
                shuffle=(mode == 'train'),
                drop_last=False, num_workers=num_workers, pin_memory=True)
        for X, y_hat in dl:
            print('X:{0}; y:{1};'.format(X, y_hat))