#
import numpy as np
import torch

class ChpA01E01(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print('线性回归 adam')
        #self.lnrn_plain()
        #self.lnrn_sgd()
        #self.lnrn_adam()
        self.lnrn_adam_mse()

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