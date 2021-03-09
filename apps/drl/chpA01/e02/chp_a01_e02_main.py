#
from torch.utils.data import DataLoader
from apps.drl.chpA01.e02.covid2019_ds import Covid2019Ds
from apps.drl.chpA01.e02.covid2019_model import Covid2019Model

class ChpA01E02Main(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print('新冠预测作业')
        mode = 'train'
        train_ds = Covid2019Ds('./data/lhy/hw1/covid.train.csv', mode=mode)
        batch_size = 32
        num_workers = 4
        train_dl = DataLoader(train_ds, batch_size, 
                shuffle=(mode == 'train'),
                drop_last=False, num_workers=num_workers, pin_memory=True)
        model = Covid2019Model(train_ds.dim)
        for X, y_hat in train_dl:
            print('X.shape:{0}; y={1};'.format(X.shape, y_hat))
            y = model(X)
            print(y)
            break
