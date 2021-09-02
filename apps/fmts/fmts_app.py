# 金融市场交易系统（Financial Market Trading System）
from argparse import ArgumentParser
import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from apps.fmts.conf.app_config import AppConfig
from apps.fmts.ds.ohlcv_dataset import OhlcvDataset
from apps.fmts.ds.ohlcv_processor import OhlcvProcessor
from apps.fmts.ann.fmts_transformer import FmtsTransformer

class FmtsApp(object):
    def __init__(self):
        self.name = 'apps.fmts.fmts_app.FmtsApp'
        self.ckpt_file = './work/fmts_v1.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def startup(self, args={}):
        print('金融市场交易系统 v0.0.8')
        self.train()

    def train(self):
        cmd_args = self.parse_args()
        stock_symbol = 'sh600260'
        batch_size = cmd_args.batch_size
        NUM_CLS = 3
        cmd_args.embedding_size = 5
        seq_length = 11
        cmd_args.num_heads = 4
        cmd_args.depth = 6 # 原始值为2
        train_iter, test_iter = self.load_stock_dataset(stock_symbol, batch_size)
        cmd_args.num_heads = 8
        model = FmtsTransformer(emb=cmd_args.embedding_size, heads=cmd_args.num_heads, depth=cmd_args.depth, \
                    seq_length=seq_length, num_tokens=cmd_args.vocab_size, num_classes=NUM_CLS, \
                    max_pool=cmd_args.max_pool)
        model.to(self.device)
        opt = torch.optim.Adam(lr=cmd_args.lr, params=model.parameters())
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (cmd_args.lr_warmup / cmd_args.batch_size), 1.0))
        if cmd_args.continue_train:
            e, model_dict, optimizer_dict = self.load_ckpt(self.ckpt_file)
            model.load_state_dict(model_dict)
            opt.load_state_dict(optimizer_dict)
        # training loop
        cmd_args.num_epochs = 3
        seen = 0
        # early stopping参数
        best_acc = -1
        acc_up = 0.0
        min_acc_up = 0.000001 # 识别为精度提高的最小阈值
        non_acc_up_epochs = 0 # 目前多少个epoch精度未提高
        max_no_acc_up_epochs = 50 # 如果精度在这些epoch后还没提高则终止训练过程
        for epoch in range(cmd_args.num_epochs):
            print(f'\n epoch {epoch}')
            model.train(True)
            for batch in tqdm.tqdm(train_iter):
                opt.zero_grad()
                X, y = self.get_stock_batch_sample(batch, batch_size, cmd_args.embedding_size)
                y_hat = model(X)
                loss = F.nll_loss(y_hat, y)
                loss.backward()
                # clip gradients
                # - If the total gradient vector has a length > 1, we clip it back down to 1.
                if cmd_args.gradient_clipping > 0.0:
                    nn.utils.clip_grad_norm_(model.parameters(), cmd_args.gradient_clipping)
                opt.step()
                sch.step()
                seen += X.size(0)
            with torch.no_grad():
                model.train(False)
                tot, cor= 0.0, 0.0 
                for batch in tqdm.tqdm(test_iter):
                    X, y = self.get_stock_batch_sample(batch, batch_size, cmd_args.embedding_size)
                    y_hat = model(X).argmax(dim=1)
                    tot += float(X.size(0))
                    cor += float((y == y_hat).sum().item())
                acc = cor / tot
                # 获取当前最佳测试集精度，并保存对应的模型
                if best_acc < acc:
                    acc_up = acc - best_acc
                    if acc_up > min_acc_up:
                        best_acc = acc
                        non_acc_up_epochs = 0
                        print('保存模型参数')
                        self.save_ckpt(self.ckpt_file, epoch, model, opt)
                else:
                    non_acc_up_epochs += 1
                    if non_acc_up_epochs > max_no_acc_up_epochs:
                        print('模型已经处于饱合状态，停止训练过程')
                        break
                print(f'-- {"test" if cmd_args.final else "validation"} accuracy {acc:.3}')
        print('^_^  v0.0.8  ^_^')

    def get_stock_batch_sample(self, batch, batch_size, embedding_size):
        X = batch[0].view(batch_size, -1, embedding_size).float().to(self.device)
        y = batch[1].long().to(self.device)
        return X, y

    def save_ckpt(self, ckpt_file, epoch, model, optimizer):
        data = {
            'epoch': epoch+1,
            'model_dict': model.state_dict(),
            'optimizer_dict': optimizer.state_dict()
        }
        torch.save(data, ckpt_file)

    def load_ckpt(self, ckpt_file):
        ckpt_obj = torch.load(ckpt_file)
        return ckpt_obj['epoch'], ckpt_obj['model_dict'], ckpt_obj['optimizer_dict']

    def load_stock_dataset(self, stock_symbol, batch_size):
        '''
        获取股票数据集
        '''
        X, y, info = OhlcvProcessor.get_ds_raw_data(stock_symbol, window_size=10, forward_size=100)
        # 生成训练数据集
        train_persent = 0.9
        train_test_sep = int(X.shape[0] * train_persent)
        X_train = X[:]
        y_train = y[:]
        info_train = info[:]
        train_ds = OhlcvDataset(X_train, y_train, info_train)
        train_loader = DataLoader(
            train_ds,
            batch_size = batch_size,
            num_workers = 0,
            shuffle = True,
            drop_last = True
        )
        train_iter = train_loader
        # 生成测试数据集
        X_test = X[train_test_sep:]
        y_test = y[train_test_sep:]
        info_test = info[train_test_sep:]
        test_ds = OhlcvDataset(X_test, y_test, info_test)
        test_loader = DataLoader(
            test_ds,
            batch_size = batch_size,
            num_workers = 0,
            shuffle = True,
            drop_last = True
        )
        test_iter = test_loader
        return train_iter, test_iter

    def parse_args(self):
        '''
        获取命令行参数并设置缺省值
        '''
        parser = ArgumentParser()
        parser.add_argument('--mode', dest='run_mode', default='bizdmrl', type=str)
        parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)
        parser.add_argument('-c', dest='continue_train', default=False, help='continue training process from ckpt', type=bool)
        parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)
        parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)
        parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')
        parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")
        parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")
        parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)
        parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)
        parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)
        parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)
        parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)
        parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)
        parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10_000, type=int)
        parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)
        return parser.parse_args()
