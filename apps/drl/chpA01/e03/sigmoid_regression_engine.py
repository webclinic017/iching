#
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from apps.drl.chpA01.e03.sigmoid_regression_ds import SigmoidRegressionDs
from apps.drl.chpA01.e03.sigmoid_regression_model import SigmoidRegressionModel

class SigmoidRegressionEngine(object):
    def __init__(self):
        self.name = ''
        self.model_file = './work/srm.ckpt'

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
        num_workers = 0
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
        
        # define the model
        device = self.get_exec_device()
        model = SigmoidRegressionModel().to(device)
        # define the loss function
        criterion = torch.nn.MSELoss()
        # define optimization method
        #learning_params = model.parameters() # 需要epochs=100才能收敛
        learning_params = []
        for k, v in model.named_parameters():
            if k == 'w001':
                learning_params.append({'params': v, 'lr': 0.02})
            elif k == 'b001':
                learning_params.append({'params': v, 'lr': 0.02})
            elif k == 'c001':
                learning_params.append({'params': v, 'lr': 0.02})
        optimizer = torch.optim.Adam(learning_params, lr=0.001)
        epochs = 1000
        best_loss = 10000.0
        unimproved_loop = 0
        improved_threshold = 0.000000001
        max_unimproved_loop = 5
        train_done = False
        for epoch in range(epochs):
            model.train()
            for X, y_hat in train_dl:
                optimizer.zero_grad()
                X, y_hat = X.to(device), y_hat.to(device)
                y = model(X)
                loss = criterion(y, y_hat)
                lossv = 0.0
                for Xv, yv_hat in valid_dl:
                    with torch.no_grad():
                        Xv, yv_hat = Xv.to(device), yv_hat.to(device)
                        yv = model(Xv)
                        lossv += criterion(yv, yv_hat)
                lossv /= valid_count
                if lossv < best_loss:
                    # save the model
                    torch.save(model.state_dict(), self.model_file)
                    if lossv < best_loss - improved_threshold:
                        unimproved_loop = 0
                    else:
                        unimproved_loop += 1
                    best_loss = lossv
                if unimproved_loop >= max_unimproved_loop:
                    train_done = True
                    break
                # early stopping处理
                loss.backward()
                optimizer.step()
                print('{0}: w={1}; b={2}; loss={3};'.format(epoch, model.w001, model.b001, loss))
            if train_done:
                break
        # 模型验证
        test_loss = 0
        for X, y_hat in test_dl:
            X, y_hat = X.to(device), y_hat.to(device)
            with torch.no_grad():
                y = model(X)
                test_loss += criterion(y, y_hat)
        test_loss /= len(test_ds)
        print('测试集上代价函数值：{0};'.format(test_loss))
        # 载入模型
        ckpt = torch.load(self.model_file)
        m1 = SigmoidRegressionModel()
        print('初始值：w={0}; b={1};'.format(m1.w001, m1.b001))
        m1.load_state_dict(ckpt)
        print('载入值：w={0}; b={1}; c={2};'.format(m1.w001, m1.b001, m1.c001))
        X = np.linspace(-25, 29, 300)
        X_ = torch.from_numpy(X)
        y_ = m1(X_)
        y = y_.detach().numpy()
        self.draw_curves(ds.X, ds.y, X, y)
        
    def get_exec_device(self):
        gpu_num = torch.cuda.device_count()
        for gi in range(gpu_num):
            print(torch.cuda.get_device_name(gi))
        pref_gi = 0
        if torch.cuda.is_available():
            if pref_gi is not None:
                device = 'cuda:{0}'.format(pref_gi)
            else:
                device = 'cuda'
        else:
            device = 'cpu'
        #device1 = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def draw_curves(self, X, y, X_, y_):
        plt.figure(figsize=(5, 5))
        plt.plot(X, y, c='r')
        plt.plot(X_, y_, c='b')
        plt.xlim(-5.5, 9.5)
        plt.ylim(0.0, 3.5)
        plt.show()