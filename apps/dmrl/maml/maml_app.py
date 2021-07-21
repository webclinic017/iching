#
import sys
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
        mode = 2
        stock_symbols = ['sh600260', 'sh600487', 'sh600728']
        if 1 == mode:
            self.prepare_ds(stock_symbols)
        elif 2 == mode:
            self.train()
        elif 3 == mode:
            self.evaluate_on_test_ds()
        elif 100000 == mode:
            self.exp()

    def get_stock_ds(self, stock_symbol):
        pass

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
        max_epoch = 10 #40
        eval_batches = 20
        #Xs, ys = self.load_ds_from_txt(stock_symbols)
        ds = AksDs(target_stock, n_way=n_way, k_shot=k_shot, q_query=q_query)
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
        meta_model = MamlModel(self.in_size, n_way).to(self.device)
        optimizer = torch.optim.Adam(meta_model.parameters(), lr = meta_lr)
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        for epoch in range(max_epoch):
            print("Epoch %d" %(epoch))
            train_meta_loss = []
            train_acc = []
            for step in tqdm(range(len(train_loader) // 
                        (meta_batch_size))): # 這裡的 step 是一次 meta-gradinet update step
                x, y, train_iter = self.get_meta_batch(
                    meta_batch_size, k_shot, q_query, 
                    train_loader, train_iter
                )
                meta_loss, acc = self.train_batch(
                    meta_model, optimizer, x, y, n_way, 
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
                x, y, val_iter = self.get_meta_batch(
                    eval_batches, k_shot, q_query, 
                    val_loader, val_iter
                )
                _, acc = self.train_batch(
                    meta_model, optimizer, x, y, n_way, 
                    k_shot, q_query, loss_fn, 
                    inner_train_steps = 3, train = False
                ) # testing時，我們更新三次 inner-step
                val_acc.append(acc)
            print("  Validation accuracy: ", np.mean(val_acc))
        print('train is OK!')
        torch.save(meta_model.state_dict(), self.chpt_file)
        self.evaluate_on_test_ds(test_loader, test_iter)


    def evaluate_on_test_ds(self, test_loader, test_iter):
        n_way = 3
        k_shot = 16
        q_query = 4
        test_batches = 4
        meta_lr = 0.001
        '''
        test_loader = DataLoader(OmniglotDs(test_data_path, n_way, k_shot, q_query),
                                batch_size = n_way,
                                num_workers = 8,
                                shuffle = True,
                                drop_last = True)
        test_iter = iter(test_loader)
        '''
        test_acc = []
        meta_model = MamlModel(self.in_size, n_way).to(self.device)
        meta_model.load_state_dict(torch.load(self.chpt_file))
        optimizer = torch.optim.Adam(meta_model.parameters(), lr = meta_lr)
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        for test_step in tqdm(range(len(test_loader) // (test_batches))):
            x, y, val_iter = self.get_meta_batch(test_batches, k_shot, q_query, test_loader, test_iter)
            _, acc = self.train_batch(meta_model, optimizer, x, y, n_way, k_shot, q_query, loss_fn, inner_train_steps = 3, train = False) # testing時，我們更新三次 inner-step
            test_acc.append(acc)
        print("  Testing accuracy: ", np.mean(test_acc))

    def train_batch(self, model, optimizer, x, y, n_way, k_shot, 
                q_query, loss_fn, inner_train_steps= 1, 
                inner_lr = 0.4, train = True):
        """
        Args:
        x is the input omniglot images for a meta_step, shape = [batch_size, n_way * (k_shot + q_query), 1, 28, 28]
        n_way: 每個分類的 task 要有幾個 class
        k_shot: 每個類別在 training 的時候會有多少張照片
        q_query: 在 testing 時，每個類別會用多少張照片 update
        """
        criterion = loss_fn
        task_loss = [] # 這裡面之後會放入每個 task 的 loss 
        task_acc = []  # 這裡面之後會放入每個 task 的 loss 
        for meta_batch, meta_batch_y in zip(x, y):
            train_set = meta_batch[:n_way*k_shot] # train_set 是我們拿來 update inner loop 參數的 data
            val_set = meta_batch[n_way*k_shot:]   # val_set 是我們拿來 update outer loop 參數的 data
            fast_weights = OrderedDict(model.named_parameters()) # 在 inner loop update 參數時，我們不能動到實際參數，因此用 fast_weights 來儲存新的參數 θ'
            for inner_step in range(inner_train_steps): # 這個 for loop 是 Algorithm2 的 line 7~8
                                                # 實際上我們 inner loop 只有 update 一次 gradients，不過某些 task 可能會需要多次 update inner loop 的 θ'，
                                                # 所以我們還是用 for loop 來寫
                train_label = self.create_label(n_way, k_shot).to(self.device)
                logits = model.functional_forward(train_set, fast_weights)
                loss = criterion(logits, train_label)
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph = True) # 這裡是要計算出 loss 對 θ 的微分 (∇loss) 
                #fast_weights = OrderedDict((name, param - inner_lr * grad)
                #                  for ((name, param), grad) in zip(fast_weights.items(), grads)) # 這裡是用剛剛算出的 ∇loss 來 update θ 變成 θ'
                fast_weights = OrderedDict((name, param) if name in self.fixed_weights else (name, param - inner_lr * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))
            val_label = self.create_label(n_way, q_query).to(self.device)
            logits = model.functional_forward(val_set, fast_weights) # 這裡用 val_set 和 θ' 算 logit
            loss = criterion(logits, val_label)                      # 這裡用 val_set 和 θ' 算 loss
            task_loss.append(loss)                                   # 把這個 task 的 loss 丟進 task_loss 裡面
            acc = np.asarray([torch.argmax(logits, -1).cpu().numpy() == val_label.cpu().numpy()]).mean() # 算 accuracy
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
        data = []
        y = torch.tensor([])
        for _ in range(meta_batch_size):
            try:
                task_data, y = iterator.next()  # 一筆 task_data 就是一個 task 裡面的 data，大小是 [n_way, k_shot+q_query, 1, 28, 28]
            except StopIteration:
                iterator = iter(data_loader)
                task_data, y = iterator.next()
            train_data = task_data[:, :k_shot].reshape(-1, self.in_size)
            val_data = task_data[:, k_shot:].reshape(-1, self.in_size)
            task_data = torch.cat((train_data, val_data), 0)
            data.append(task_data)
        return torch.stack(data).float().to(self.device), y, iterator


































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