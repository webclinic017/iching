#
import sys
from matplotlib.pyplot import axis
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader #, Dataset
from collections import OrderedDict
from apps.dmrl.maml.omniglot_ds import OmniglotDs
from apps.dmrl.maml.maml_model import MamlModel
from apps.dmrl.maml.aks_ds import AksDs
from apps.dmrl.maml.app_config import AppConfig

class MamlApp(object):
    def __init__(self):
        self.name = 'apps.dmrl.maml.MamlApp'
        self.chpt_file = './work/maml.pkl'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fixed_weights = [] # 需要固定的权值名称列表
        self.in_size = 50 # 每个行情包括之前10天（包括今天），每天有开盘、最高、最低、收盘、交易量这5个值

    def startup(self):
        print('MAML算法 v0.0.2')
        mode = 4
        stock_symbols = ['sh600260', 'sh600487', 'sh600728']
        if 1 == mode:
            self.prepare_ds(stock_symbols)
        elif 2 == mode:
            self.train()
        elif 3 == mode:
            self.evaluate_on_test_ds()
        elif 4 == mode:
            self.run_adapt_process()
        elif 100000 == mode:
            self.exp()

    def train(self):
        ref_stocks = ['sh600487', 'sh600728']
        target_stock = 'sh600260'
        torch.cuda.set_device(0)
        n_way = 3
        k_shot = 16
        q_query = 4
        inner_train_steps = 1
        inner_lr = 0.4
        meta_lr = 0.001
        meta_batch_size = 8 #32
        max_epoch = 2 #40
        eval_batches = 20
        #Xs, ys = self.load_ds_from_txt(stock_symbols)
        train_loaders, train_iters = [], []
        val_loaders, val_iters = [], []
        test_loaders, test_iters = [], []

        self.load_stock_datas(target_stock, n_way, k_shot, q_query, \
                    train_loaders, train_iters, \
                    val_loaders, val_iters, \
                    test_loaders, test_iters)
        for stock_symbol in ref_stocks:
            self.load_stock_datas(stock_symbol, n_way, k_shot, q_query, \
                    train_loaders, train_iters, \
                    val_loaders, val_iters, \
                    test_loaders, test_iters)
        train_loader = train_loaders[0]
        train_iter = train_iters[0]
        val_loader = val_loaders[0]
        val_iter = val_iters[0]
        test_loader = test_loaders[0]
        test_iter = test_iters[0]
        stock_cnt = len(train_loaders) # 股票数量

        meta_model = MamlModel(self.in_size, n_way).to(self.device)
        optimizer = torch.optim.Adam(meta_model.parameters(), lr = meta_lr)
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        for epoch in range(max_epoch):
            print("Epoch %d" %(epoch))
            train_meta_loss = []
            train_acc = []
            for step in tqdm(range(len(train_loader) // 
                        (meta_batch_size))): # 這裡的 step 是一次 meta-gradinet update step
                xs, ys = [], []
                for idx in range(stock_cnt):
                    x, y, train_iters[idx] = self.get_meta_batch(
                        meta_batch_size, k_shot, q_query, 
                        train_loaders[idx], train_iters[idx]
                    )
                    xs.append(x)
                    ys.append(y)
                meta_loss, acc = self.train_batch(
                    meta_model, optimizer, xs, ys, n_way, 
                    k_shot, q_query, loss_fn
                )
                train_meta_loss.append(meta_loss.item())
                train_acc.append(acc)
            print("  Loss    : ", np.mean(train_meta_loss))
            print("  Accuracy: ", np.mean(train_acc))
            # 每個 epoch 結束後，看看 validation accuracy 如何  
            # 助教並沒有做 early stopping，同學如果覺得有需要是可以做的 
            val_acc = []
            for eval_step in tqdm(range(len(val_loader) // 
                        (eval_batches))):
                xs, ys = [], []
                for idx in range(stock_cnt):
                    x, y, val_iters[idx] = self.get_meta_batch(
                        eval_batches, k_shot, q_query, 
                        val_loaders[idx], val_iters[idx]
                    )
                    xs.append(x)
                    ys.append(y)
                _, acc = self.train_batch(
                    meta_model, optimizer, xs, ys, n_way, 
                    k_shot, q_query, loss_fn, 
                    inner_train_steps = 3, train = False
                ) # testing時，我們更新三次 inner-step
                val_acc.append(acc)
            print("  Validation accuracy: ", np.mean(val_acc))
        print('train is OK!')
        torch.save(meta_model.state_dict(), self.chpt_file)
        self.evaluate_on_test_ds(target_stock)


    def evaluate_on_test_ds(self, stock_symbol):
        n_way = 3
        k_shot = 16
        q_query = 4
        test_batches = 4
        meta_lr = 0.001
        test_acc = []        
        test_ds = AksDs(stock_symbol, n_way=n_way, k_shot=k_shot, q_query=q_query, ds_mode=AksDs.DS_MODE_TEST, train_rate=0.95, val_rate=0.0, test_rate=0.05)
        test_loader = DataLoader(
            test_ds,
            batch_size = n_way,
            num_workers = 0,
            shuffle = True,
            drop_last = True
        )
        test_iter = iter(test_loader)
        meta_model = MamlModel(self.in_size, n_way).to(self.device)
        meta_model.load_state_dict(torch.load(self.chpt_file))
        optimizer = torch.optim.Adam(meta_model.parameters(), lr = meta_lr)
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        for test_step in tqdm(range(len(test_loader) // (test_batches))):
            x, y, val_iter = self.get_meta_batch(test_batches, k_shot, q_query, test_loader, test_iter)
            xs = []
            ys = []
            xs.append(x)
            ys.append(y)
            _, acc = self.train_batch(meta_model, optimizer, xs, ys, n_way, k_shot, q_query, loss_fn, inner_train_steps = 3, train = False) # testing時，我們更新三次 inner-step
            test_acc.append(acc)
        print("  Testing accuracy: ", np.mean(test_acc))

    def run_adapt_process(self):
        stock_symbol = 'sh600260'
        n_way = 3
        k_shot = 16
        q_query = 4
        test_batches = 4
        meta_lr = 0.001
        meta_model = MamlModel(self.in_size, n_way).to(self.device)
        meta_model.load_state_dict(torch.load(self.chpt_file))
        optimizer = torch.optim.Adam(meta_model.parameters(), lr = meta_lr)
        loss_fn = nn.CrossEntropyLoss().to(self.device)  
        test_ds = AksDs(stock_symbol, n_way=n_way, k_shot=k_shot, q_query=q_query, ds_mode=AksDs.DS_MODE_TEST, train_rate=0.95, val_rate=0.0, test_rate=0.03)
        test_loader = DataLoader(
            test_ds,
            batch_size = n_way,
            num_workers = 0,
            shuffle = True,
            drop_last = True
        )
        test_iter = iter(test_loader)
        for test_step in tqdm(range(len(test_loader) // (test_batches))):
            x, y, test_iter = self.get_meta_batch(test_batches, k_shot, q_query, test_loader, test_iter)
            self.adapt_to_new_env(meta_model, optimizer, loss_fn, x, y, 3)
        # 模拟实际运行过程
        t1_ds = AksDs(stock_symbol, n_way=n_way, k_shot=k_shot, q_query=q_query, ds_mode=AksDs.DS_MODE_TEST, train_rate=0.98, val_rate=0.0, test_rate=0.02)
        t1_loader = DataLoader(
            t1_ds,
            batch_size = n_way,
            num_workers = 0,
            shuffle = True,
            drop_last = True
        )
        t1_iter = iter(t1_loader)
        accs = np.array([])
        for X_raw, y_raw in t1_iter:
            X = X_raw.float().reshape((-1, X_raw.shape[-1])).to(self.device)
            y = y_raw.long().reshape((-1, ))
            y_hat = self.run_predict(meta_model, X)
            acc = np.asarray([y_hat.cpu().numpy() == y.cpu().numpy()]).mean()
            print('最终精度：{0};'.format(acc))
            accs = np.append(accs, acc)
        score = accs.mean()
        print('score={0};'.format(score))
            

    def adapt_to_new_env(self, model, optimizer, loss_fn, ds_x, ds_y, loop_num=1):
        '''
        已经训练好的模型，使用少量新数据，训练出一个足够好的模型用于新环境
        '''
        criterion = loss_fn
        model.train()
        for _ in range(loop_num):
            for x, y in zip(ds_x, ds_y):
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def run_predict(self, model, x):
        logits = model(x)
        return torch.argmax(logits, -1)

    def train_batch(self, model, optimizer, xs, ys, n_way, k_shot, 
                q_query, loss_fn, inner_train_steps= 1, 
                inner_lr = 0.4, train = True):
        """
        Args:
        x is the input omniglot images for a meta_step, shape = [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
        n_way: 每個分類的 task 要有幾個 class
        k_shot: 每個類別在 training 的時候會有多少張照片
        q_query: 在 testing 時，每個類別會用多少張照片 update
        """
        task_cnt = len(xs)
        outer_loop = xs[0].shape[0]
        criterion = loss_fn
        task_loss = [] # 這裡面之後會放入每個 task 的 loss 
        task_acc = []  # 這裡面之後會放入每個 task 的 loss 
        for oi in range(outer_loop):
            theta_primes = []
            for ti in range(task_cnt):
                meta_batch = xs[ti][oi]
                meta_batch_y = ys[ti][oi]
                support_set_x = meta_batch[:n_way*k_shot] # train_set 是我們拿來 update inner loop 參數的 data
                support_set_y = meta_batch_y[:n_way*k_shot]
                query_set_x = meta_batch[n_way*k_shot:]   # val_set 是我們拿來 update outer loop 參數的 data
                query_set_y = meta_batch_y[n_way*k_shot:]
                theta_prime_i = OrderedDict(model.named_parameters())
                for inner_step in range(inner_train_steps):
                    logits = model.functional_forward(support_set_x, theta_prime_i)
                    loss = criterion(logits, support_set_y)
                    grads = torch.autograd.grad(loss, theta_prime_i.values(), create_graph = True)
                    theta_prime_i = OrderedDict((name, param) if name in self.fixed_weights else (name, param - inner_lr * grad) \
                                for ((name, param), grad) in zip(theta_prime_i.items(), grads))
                theta_primes.append(theta_prime_i)
            #
            loss = 0.0
            task_accs = np.array([])
            for i in range(task_cnt):
                logits = model.functional_forward(query_set_x, theta_primes[i])
                task_accs = np.append(task_accs, np.asarray([torch.argmax(logits, -1).cpu().numpy() == query_set_y.cpu().numpy()]).mean())
                loss += criterion(logits, query_set_y)
                grads += torch.autograd.grad(loss, theta_primes[i].values(), create_graph = True)
            task_loss.append(loss)                                   # 把這個 task 的 loss 丟進 task_loss 裡面
            acc = task_accs.mean() # 算 accuracy
            task_acc.append(acc)
        model.train()
        optimizer.zero_grad()
        meta_batch_loss = torch.stack(task_loss).mean() # 我們要用一整個 batch 的 loss 來 update θ (不是 θ')
        if train:
            meta_batch_loss.backward()
            optimizer.step()
        task_acc = np.mean(task_acc)
        return meta_batch_loss, task_acc

    def get_meta_batch(self, meta_batch_size, k_shot, q_query, data_loader, iterator):
        data_x = []
        data_y = []
        y = torch.tensor([])
        for _ in range(meta_batch_size):
            try:
                task_data_x, task_data_y = iterator.next()  # 一筆 task_data 就是一個 task 裡面的 data，大小是 [n_way, k_shot+q_query, 1, 28, 28]
            except StopIteration:
                iterator = iter(data_loader)
                task_data_x, task_data_y = iterator.next()
            train_data_x = task_data_x[:, :k_shot].reshape(-1, self.in_size)
            train_data_y = task_data_y[:, :k_shot].reshape(-1)
            val_data_x = task_data_x[:, k_shot:].reshape(-1, self.in_size)
            val_data_y = task_data_y[:, k_shot:].reshape(-1)
            task_data_x = torch.cat((train_data_x, val_data_x), 0)
            task_data_y = torch.cat((train_data_y, val_data_y), 0)
            data_x.append(task_data_x)
            data_y.append(task_data_y)
        return torch.stack(data_x).float().to(self.device), torch.stack(data_y).long().to(self.device), iterator


    def get_stock_ds(self, stock_symbol, n_way, k_shot, q_query):
        '''
        ds = AksDs(stock_symbol, n_way=n_way, k_shot=k_shot, q_query=q_query)
        print('ds_obj: size={0};'.format(len(ds)))
        cnt = len(ds)
        raw_train_cnt = int(cnt * 0.95)
        test_cnt = cnt - raw_train_cnt
        train_cnt = int(raw_train_cnt * 0.95)
        val_cnt = raw_train_cnt - train_cnt
        #raw_train_ds, test_ds = torch.utils.data.random_split(ds, [raw_train_cnt, test_cnt])
        #train_ds, val_ds = torch.utils.data.random_split(raw_train_ds, [train_cnt, val_cnt])
        train_ds, test_ds = torch.utils.data.random_split(ds, [raw_train_cnt, test_cnt])
        val_ds = test_ds
        '''
        train_ds = AksDs(stock_symbol, n_way=n_way, k_shot=k_shot, q_query=q_query, ds_mode=AksDs.DS_MODE_TRAIN, train_rate=0.95)
        test_ds = AksDs(stock_symbol, n_way=n_way, k_shot=k_shot, q_query=q_query, ds_mode=AksDs.DS_MODE_TEST, train_rate=0.95, val_rate=0.0, test_rate=0.05)
        val_ds = test_ds
        train_loader = DataLoader(train_ds,
            batch_size = n_way, # 多少个类别
            num_workers = 0, # 4：原值，但是只能取0否则报异常
            shuffle = True,
            drop_last = True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size = n_way,
            num_workers = 0,
            shuffle = False,
            drop_last = True
        )
        test_loader = DataLoader(
            test_ds,
            batch_size = n_way,
            num_workers = 0,
            shuffle = True,
            drop_last = True
        )
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)
        test_iter = iter(test_loader)
        return train_loader, train_iter, val_loader, val_iter, test_loader, test_iter

    def load_stock_datas(self, stock_symbol, \
                    n_way, k_shot, q_query, \
                    train_loaders, train_iters, \
                        val_loaders, val_iters, \
                            test_loaders, test_iters):
        train_loader, train_iter, val_loader, val_iter, test_loader, test_iter = \
                    self.get_stock_ds(stock_symbol, n_way, k_shot, q_query)
        train_loaders.append(train_loader)
        train_iters.append(train_iter)
        val_loaders.append(val_loader)
        val_iters.append(val_iter)
        test_loaders.append(test_loader)
        test_iters.append(test_iter)































    def exp(self):
        print('MAML算法试验代码')

    def exp002(self):
        '''
        从文件中读出数据集
        '''
        stock1_symbol = 'sh600260'
        ds = AksDs()
        X1 = ds.load_X_from_csv(stock1_symbol)
        y1 = ds.load_y_from_csv(stock1_symbol)
        print('X1: {0};'.format(X1.shape))
        print('y1: {0}; dtype={1};'.format(y1.shape, y1.dtype))
        ###############################################################################
        #################### 程序结束标志 #######################################
        ###############################################################################
        print('^_^ The End ^_^')

    def exp001(self):
        '''
        以收盘价为例，画实时折线图
        '''
        stock1_symbol = 'sh600260'
        ds = AksDs()
        Xs = []
        ys = []
        X1, y1 = ds.generate_stock_ds(stock1_symbol, draw_line=True)
        Xs.append(X1)
        ys.append(y1)
        ###############################################################################
        #################### 程序结束标志 #######################################
        ###############################################################################
        print('^_^ The End ^_^')   










    def create_label(self, n_way, k_shot):
        return torch.arange(n_way).repeat_interleave(k_shot).long()

    

    def prepare_ds(self, stock_symbols):
        '''
        生成数据集
        '''
        ds = AksDs()
        for stock_symbol in stock_symbols:
            X, y = ds.generate_stock_ds(stock_symbol, draw_line=False)
            X_file = './data/aks_ds/{0}_X.txt'.format(stock_symbol)
            np.savetxt(X_file, X, delimiter=',', newline='\n', encoding='utf-8')
            y_file = './data/aks_ds/{0}_y.txt'.format(stock_symbol)
            np.savetxt(y_file, y, delimiter=',', newline='\n', encoding='utf-8')
            print('X: {0}; y: {1};'.format(X.shape, y.shape))

    def load_ds_from_txt(self, stock_symbols):
        '''
        从文本文件中读出行情数据集，所有股票以字典形式返回
        '''
        Xs = {}
        ys = {}
        for stock_symbol in stock_symbols:
            X_file = './data/aks_ds/{0}_X.txt'.format(stock_symbol)
            X = np.loadtxt(X_file, delimiter=',', encoding='utf-8')
            Xs[stock_symbol] = X
            y_file = './data/aks_ds/{0}_y.txt'.format(stock_symbol)
            y = np.loadtxt(y_file, delimiter=',', encoding='utf-8')
            ys[stock_symbol] = y
        return Xs, ys



    '''
    # 创新实验性代码
    '''

    def set_epoch_fixed_weights(self, epoch):
        if epoch == 1:
            self.fixed_weights.append('conv1.0.weight')
            self.fixed_weights.append('conv1.0.bias')
            self.fixed_weights.append('conv1.1.weight')
            self.fixed_weights.append('conv1.1.bias')
        elif epoch == 2:
            self.fixed_weights.append('conv2.0.weight')
            self.fixed_weights.append('conv2.0.bias')
            self.fixed_weights.append('conv2.1.weight')
            self.fixed_weights.append('conv2.1.bias')
        elif epoch == 3:
            self.fixed_weights.append('conv3.0.weight')
            self.fixed_weights.append('conv3.0.bias')
            self.fixed_weights.append('conv3.1.weight')
            self.fixed_weights.append('conv3.1.bias')
        elif epoch == 4:
            self.fixed_weights.append('conv4.0.weight')
            self.fixed_weights.append('conv4.0.bias')
            self.fixed_weights.append('conv4.1.weight')
            self.fixed_weights.append('conv4.1.bias')