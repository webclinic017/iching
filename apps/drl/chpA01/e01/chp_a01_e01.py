#
import numpy as np
import torch
from torch.utils.data import DataLoader
from apps.drl.chpA01.e01.chp_a01_e01_ds import ChpA01E01Ds
from apps.drl.chpA01.e01.chp_a01_e01_model import ChpA01E01Model

class ChpA01E01(object):
    def __init__(self):
        self.name = ''
        self.model_file = './work/lnrn.pt'

    def startup(self, args={}):
        print('线性回归 adam')
        #self.lnrn_plain()
        #self.lnrn_sgd()
        #self.lnrn_adam()
        #self.lnrn_adam_mse()
        #self.lnrn_with_ds()
        #self.lnrn_with_model()
        #self.lnrn_gpu()
        #self.lnrn_eval()
        #self.lnrn_save_load()
        #self.lnrn_ds_split() # train and valid split

    def lnrn_ds_split():
        pass

    def lnrn_eval(self):
        # load dataset
        ds = ChpA01E01Ds(num=1000)
        batch_size = 10
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        # define the model
        device = self.get_exec_device()
        model = ChpA01E01Model().to(device)
        # define the loss function
        criterion = torch.nn.MSELoss()
        # define optimization method
        #learning_params = model.parameters() # 需要epochs=100才能收敛
        learning_params = []
        for k, v in model.named_parameters():
            if k == 'w001':
                learning_params.append({'params': v, 'lr': 0.01})
            elif k == 'b001':
                learning_params.append({'params': v, 'lr': 0.1})
        optimizer = torch.optim.Adam(learning_params, lr=0.001)
        epochs = 10
        for epoch in range(epochs):
            model.train()
            for X, y_hat in dl:
                optimizer.zero_grad()
                X, y_hat = X.to(device), y_hat.to(device)
                y = model(X)
                loss = criterion(y, y_hat)
                loss.backward()
                optimizer.step()
                print('{0}: w={1}; b={2}; loss={3};'.format(epoch, model.w001, model.b001, loss))
        # 模型验证
        test_num = 100
        test_ds = ChpA01E01Ds(num=test_num)
        model.eval()
        preds = []
        batch_size = 30
        test_dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        test_loss = 0
        for X, y_hat in test_dl:
            X, y_hat = X.to(device), y_hat.to(device)
            with torch.no_grad():
                y = model(X)
                test_loss += criterion(y, y_hat)
        test_loss /= test_num
        print('测试集上代价函数值：{0};'.format(test_loss))

    def lnrn_save_load(self):
        # load dataset
        ds = ChpA01E01Ds(num=1000)
        batch_size = 10
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        # define the model
        device = self.get_exec_device()
        model = ChpA01E01Model().to(device)
        # define the loss function
        criterion = torch.nn.MSELoss()
        # define optimization method
        #learning_params = model.parameters() # 需要epochs=100才能收敛
        learning_params = []
        for k, v in model.named_parameters():
            if k == 'w001':
                learning_params.append({'params': v, 'lr': 0.01})
            elif k == 'b001':
                learning_params.append({'params': v, 'lr': 0.1})
        optimizer = torch.optim.Adam(learning_params, lr=0.001)
        epochs = 10
        for epoch in range(epochs):
            model.train()
            for X, y_hat in dl:
                optimizer.zero_grad()
                X, y_hat = X.to(device), y_hat.to(device)
                y = model(X)
                loss = criterion(y, y_hat)
                loss.backward()
                optimizer.step()
                print('{0}: w={1}; b={2}; loss={3};'.format(epoch, model.w001, model.b001, loss))
        # 模型验证
        test_num = 100
        test_ds = ChpA01E01Ds(num=test_num)
        model.eval()
        preds = []
        batch_size = 30
        test_dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        test_loss = 0
        for X, y_hat in test_dl:
            X, y_hat = X.to(device), y_hat.to(device)
            with torch.no_grad():
                y = model(X)
                test_loss += criterion(y, y_hat)
        test_loss /= test_num
        print('测试集上代价函数值：{0};'.format(test_loss))
        print('模型保存和加载测试')
        # 保存模型
        torch.save(model.state_dict(), self.model_file)
        # 载入模型
        ckpt = torch.load(self.model_file)
        m1 = ChpA01E01Model()
        print('初始值：w={0}; b={1};'.format(m1.w001, m1.b001))
        m1.load_state_dict(ckpt)
        print('载入值：w={0}; b={1};'.format(m1.w001, m1.b001))

    def ds_exp(self):
        ds = ChpA01E01Ds(num=1000)
        batch_size = 10
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for X, y in dl:
            print('X: {0}; y: {1};'.format(X, y))
            break

    def lnrn_with_ds(self):
        # load dataset
        ds = ChpA01E01Ds(num=1000)
        batch_size = 10
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        # define the model
        w = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        # define the loss function
        criterion = torch.nn.MSELoss()
        # define optimization method
        optimizer = torch.optim.Adam([
            {'params': w, 'lr': 0.01},
            {'params': b, 'lr': 0.1}
        ], lr=0.001)
        epochs = 10
        for epoch in range(epochs):
            for X, y_hat in dl:
                optimizer.zero_grad()
                y = w * X + b
                loss = criterion(y, y_hat)
                loss.backward()
                optimizer.step()
                print('{0}: w={1}; b={2}; loss={3};'.format(epoch, w, b, loss))

    def lnrn_with_model(self):
        # load dataset
        ds = ChpA01E01Ds(num=1000)
        batch_size = 10
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        # define the model
        model = ChpA01E01Model()
        # define the loss function
        criterion = torch.nn.MSELoss()
        # define optimization method
        #learning_params = model.parameters() # 需要epochs=100才能收敛
        learning_params = []
        for k, v in model.named_parameters():
            if k == 'w001':
                learning_params.append({'params': v, 'lr': 0.01})
            elif k == 'b001':
                learning_params.append({'params': v, 'lr': 0.1})
        optimizer = torch.optim.Adam(learning_params, lr=0.001)
        epochs = 10
        for epoch in range(epochs):
            model.train()
            for X, y_hat in dl:
                optimizer.zero_grad()
                y = model(X)
                loss = criterion(y, y_hat)
                loss.backward()
                optimizer.step()
                print('{0}: w={1}; b={2}; loss={3};'.format(epoch, model.w001, model.b001, loss))

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
    
    def lnrn_gpu(self):
        # load dataset
        ds = ChpA01E01Ds(num=1000)
        batch_size = 10
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        # define the model
        device = self.get_exec_device()
        model = ChpA01E01Model().to(device)
        # define the loss function
        criterion = torch.nn.MSELoss()
        # define optimization method
        #learning_params = model.parameters() # 需要epochs=100才能收敛
        learning_params = []
        for k, v in model.named_parameters():
            if k == 'w001':
                learning_params.append({'params': v, 'lr': 0.01})
            elif k == 'b001':
                learning_params.append({'params': v, 'lr': 0.1})
        optimizer = torch.optim.Adam(learning_params, lr=0.001)
        epochs = 10
        for epoch in range(epochs):
            model.train()
            for X, y_hat in dl:
                optimizer.zero_grad()
                X, y_hat = X.to(device), y_hat.to(device)
                y = model(X)
                loss = criterion(y, y_hat)
                loss.backward()
                optimizer.step()
                print('{0}: w={1}; b={2}; loss={3};'.format(epoch, model.w001, model.b001, loss))

    def lnrn_plain(self):
        X, y_hat = self.load_ds()
        w = torch.tensor(1.0, requires_grad=True)
        w_lr = 0.01
        b = torch.tensor(0.0, requires_grad=True)
        b_lr = 0.1
        epochs = 6000
        X = torch.tensor(X)
        y_hat = torch.tensor(y_hat)
        for epoch in range(epochs):
            y = w * X + b
            tl = 0.5 * (y - y_hat)**2
            loss = tl.sum() / 1000.0
            loss.backward()
            with torch.no_grad():
                w -= w_lr * w.grad
                w.grad = torch.zeros_like(w.grad)
                b -= b_lr * b.grad
                b.grad = torch.zeros_like(b.grad)
            print('{0}: w={1}; b={2}; loss={3};'.format(epoch, w, b, loss))

    

    def lnrn_sgd(self):
        X, y_hat = self.load_ds()
        w = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        epochs = 6000
        optimizer = torch.optim.SGD([
            {'params': w, 'lr': 0.01},
            {'params': b, 'lr': 0.1}
        ], 0.001)
        X = torch.tensor(X)
        y_hat = torch.tensor(y_hat)
        for epoch in range(epochs):
            optimizer.zero_grad()
            y = w * X + b
            tl = 0.5 * (y - y_hat)**2
            loss = tl.sum() / 1000.0
            loss.backward()
            optimizer.step()
            print('{0}: w={1}; b={2}; loss={3};'.format(epoch, w, b, loss))

    

    def lnrn_adam(self):
        X, y_hat = self.load_ds()
        w = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        epochs = 6000
        optimizer = torch.optim.Adam([
            {'params': w, 'lr': 0.01},
            {'params': b, 'lr': 0.1}
        ], lr=0.001)
        X = torch.tensor(X)
        y_hat = torch.tensor(y_hat)
        for epoch in range(epochs):
            optimizer.zero_grad()
            y = w * X + b
            tl = 0.5 * (y - y_hat)**2
            loss = tl.sum() / 1000.0
            loss.backward()
            optimizer.step()
            print('{0}: w={1}; b={2}; loss={3};'.format(epoch, w, b, loss))

    

    def lnrn_adam_mse(self):
        X, y_hat = self.load_ds()
        w = torch.tensor(1.0, requires_grad=True)
        w_lr = 0.01
        b = torch.tensor(0.0, requires_grad=True)
        b_lr = 0.1
        epochs = 1000
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([
            {'params': w, 'lr': 0.01},
            {'params': b, 'lr': 0.1}
        ], lr=0.001)
        X = torch.tensor(X)
        y_hat = torch.tensor(y_hat)
        for epoch in range(epochs):
            optimizer.zero_grad()
            y = w * X + b
            loss = criterion(y, y_hat)
            loss.backward()
            optimizer.step()
            print('{0}: w={1}; b={2}; loss={3};'.format(epoch, w, b, loss))

    def load_ds(self):
        b = 1.6
        w = 0.3
        X = np.linspace(0, 1.0, num=1000)
        y = w*X + b
        return X, y