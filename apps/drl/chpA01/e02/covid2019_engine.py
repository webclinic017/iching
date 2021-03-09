#
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from apps.drl.chpA01.e02.covid2019_ds import Covid2019Ds
from apps.drl.chpA01.e02.covid2019_model import Covid2019Model

class Covid2019Engine(object):
    def __init__(self):
        myseed = 42069  # set a random seed for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(myseed)
        torch.manual_seed(myseed)

    def train(self):
        device = self.get_exec_device()
        config = self.load_config()
        train_ds, train_dl, valid_ds, valid_dl, test_ds, test_dl = \
            self.load_datas(
                train_file=config['train_file'], 
                train_batch_size=config['train_batch_size'],
                test_file=config['test_file'], 
                test_batch_size=config['test_batch_size']
            )
        model = Covid2019Model(train_ds.dim).to(device)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = getattr(torch.optim, config['optimizer'])(
                model.parameters(), **config['optim_hparams'])
        model_loss, model_loss_record = self._train(train_dl, valid_dl, model, config, device,
                    criterion=criterion, optimizer=optimizer)
        self.plot_learning_curve(model_loss_record, title='Covid2019 Learning Curve')

    def evaluate(self):
        device = self.get_exec_device()
        config = self.load_config()
        train_ds, train_dl, valid_ds, valid_dl, test_ds, test_dl = \
            self.load_datas(
                train_file=config['train_file'], 
                train_batch_size=config['train_batch_size'],
                test_file=config['test_file'], 
                test_batch_size=config['test_batch_size']
            )
        model = self.load_model(device, config)
        self.plot_pred(valid_dl, model, device) 

    def plot_pred(self, data_dl, model, device, lim=35., preds=None, targets=None):
        ''' Plot prediction of your DNN '''
        if preds is None or targets is None:
            model.eval()
            preds, targets = [], []
            for x, y in data_dl:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    pred = model(x)
                    preds.append(pred.detach().cpu())
                    targets.append(y.detach().cpu())
            preds = torch.cat(preds, dim=0).numpy()
            targets = torch.cat(targets, dim=0).numpy()
        print('preds: {0};'.format(preds))
        print('targets: {0};'.format(targets))
        figure(figsize=(5, 5))
        plt.scatter(targets, preds, c='r', alpha=0.5)
        plt.plot([-0.2, lim], [-0.2, lim], c='b')
        plt.xlim(-0.2, lim)
        plt.ylim(-0.2, lim)
        plt.xlabel('ground truth value')
        plt.ylabel('predicted value')
        plt.title('Ground Truth v.s. Prediction')
        plt.savefig('./work/evaluate.png')
        plt.show()

    def submit_result(self):
        device = self.get_exec_device()
        config = self.load_config()
        train_ds, train_dl, valid_ds, valid_dl, test_ds, test_dl = \
            self.load_datas(
                train_file=config['train_file'], 
                train_batch_size=config['train_batch_size'],
                test_file=config['test_file'], 
                test_batch_size=config['test_batch_size']
            )
        model = self.load_model(device, config)
        model.eval()                                # set model to evalutation mode
        preds = []
        for x in test_dl:                            # iterate through the dataloader
            x = x.to(device)                        # move data to device (cpu/cuda)
            with torch.no_grad():                   # disable gradient calculation
                pred = model(x)                     # forward pass (compute output)
                preds.append(pred.detach().cpu())   # collect prediction
        preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
        result_file = './work/hw1_results.csv'
        self.save_preds(preds, result_file)

    def save_preds(self, preds, file):
        ''' Save predictions to specified file '''
        print('Saving results to {}'.format(file))
        with open(file, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'tested_positive'])
            for i, p in enumerate(preds):
                writer.writerow([i, p])

    def load_model(self, device, config):
        model = Covid2019Model(Covid2019Ds.X_dim).to(device)
        ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
        model.load_state_dict(ckpt)
        return model

    def _train(self, train_dl, valid_dl, model, config, device, criterion, optimizer):
        min_mse = 1000.
        loss_record = {'train': [], 'valid': []}      # for recording training loss
        early_stop_cnt = 0
        for epoch in range(config['n_epochs']):
            model.train()
            for X, y_hat in train_dl:
                optimizer.zero_grad()
                X, y_hat = X.to(device), y_hat.to(device)
                y = model(X)
                mse_loss = criterion(y, y_hat)
                mse_loss.backward()
                optimizer.step()
                loss_record['train'].append(mse_loss.detach().cpu().item())
            valid_mse = self._evaluate(valid_dl, model, device, criterion=criterion)
            if valid_mse < min_mse:
                min_mse = valid_mse
                torch.save(model.state_dict(), config['save_path'])
                early_stop_cnt = 0
                print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            else:
                early_stop_cnt += 1
            loss_record['valid'].append(valid_mse)
            if early_stop_cnt > config['early_stop']:
                break
        print('Best loss: {0};'.format(min_mse))
        return min_mse, loss_record

    def plot_learning_curve(self, loss_record, title=''):
        ''' Plot learning curve of your DNN (train & valid loss) '''
        total_steps = len(loss_record['train'])
        x_1 = range(total_steps)
        x_2 = x_1[::len(loss_record['train']) // len(loss_record['valid'])]
        figure(figsize=(6, 4))
        plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
        plt.plot(x_2, loss_record['valid'], c='tab:cyan', label='valid')
        plt.ylim(0.0, 5.)
        plt.xlabel('Training steps')
        plt.ylabel('MSE loss')
        plt.title('Learning curve of {}'.format(title))
        plt.legend()
        plt.savefig('./work/hw1_train.png')
        plt.show()

    def _evaluate(self, data_dl, model, device, criterion):
        model.eval()                                # set model to evalutation mode
        total_loss = 0
        for X, y_hat in data_dl:                         # iterate through the dataloader
            X, y_hat = X.to(device), y_hat.to(device)       # move data to device (cpu/cuda)
            with torch.no_grad():                   # disable gradient calculation
                y = model(X)                     # forward pass (compute output)
                mse_loss = criterion(y, y_hat)  # compute loss
            total_loss += mse_loss.detach().cpu().item() * len(X)  # accumulate loss
        total_loss = total_loss / len(data_dl.dataset)              # compute averaged loss
        return total_loss

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

    def load_config(self):
        config = {
            'train_file': './data/lhy/hw1/covid.train.csv',
            'test_file': './data/lhy/hw1/covid.test.csv',
            'n_epochs': 3, #3000,                # maximum number of epochs
            'train_batch_size': 270,               # mini-batch size for dataloader
            'test_batch_size': 270,               # mini-batch size for dataloader
            'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
            'optim_hparams': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
                'lr': 0.001,                 # learning rate of SGD
                'momentum': 0.9              # momentum for SGD
            },
            'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
            'save_path': 'work/lhy001.pth'  # your model will be saved here
        }
        return config

    def load_datas(self, train_file, train_batch_size, test_file, test_batch_size):
        train_ds, train_dl = self._load_data(train_file, mode='train', batch_size=train_batch_size)
        valid_ds, valid_dl = self._load_data(train_file, mode='valid', batch_size=test_batch_size)
        test_ds, test_dl = self._load_data(test_file, mode='test', batch_size=test_batch_size)
        return train_ds, train_dl, valid_ds, valid_dl, test_ds, test_dl

    def _load_data(self, data_file, mode, batch_size):
        ds = Covid2019Ds(data_file, mode=mode)
        num_workers = 4
        dl = DataLoader(ds, batch_size, 
                shuffle=(mode == 'train'),
                drop_last=False, num_workers=num_workers, pin_memory=True)
        return ds, dl