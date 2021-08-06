# 
import math
from argparse import ArgumentParser
import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchtext
from torchtext.legacy import data, datasets, vocab
from torch.utils.data import DataLoader
from biz.dmrl.iqtt.self_attention import SelfAttention
from biz.dmrl.iqtt.iqtt_util import IqttUtil
from biz.dmrl.iqtt.iqtt_transformer import IqttTransformer
from biz.dmrl.iqtt.aks_ds import AksDs

class IqttApp(object):
    DSM_IMDB = 'imdb'
    DSM_STOCK = 'stock'

    def __init__(self):
        self.name = 'biz.dmrl.iqtt.iqtt_app.IqttApp'
        self.ds_mode = 'IMDB'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def startup(self, args={}):
        print('Iching Quantitative Trading Transformer v0.0.2')
        cmd_args = self.parse_args()
        print('command line args: {0};'.format(cmd_args))
        cmd_args.num_epochs = 1000


        
        cmd_args.num_heads = 8
        cmd_args.depth = 6



        # M1
        # train_iter, test_iter, NUM_CLS, seq_length, mx = self.load_imdb_dataset(cmd_args)
        train_iter, test_iter, NUM_CLS, seq_length, mx = self.load_stock_dataset(cmd_args)
        # M2
        # create the model
        #model = IqttTransformer(emb=cmd_args.embedding_size, heads=cmd_args.num_heads, \
        #            depth=cmd_args.depth, seq_length=mx, num_tokens=cmd_args.vocab_size, \
        #            num_classes=NUM_CLS, max_pool=cmd_args.max_pool)
        model = IqttTransformer(emb=cmd_args.embedding_size, heads=cmd_args.num_heads, depth=cmd_args.depth, \
                    seq_length=seq_length, num_tokens=cmd_args.vocab_size, num_classes=NUM_CLS, \
                    max_pool=cmd_args.max_pool, mode=IqttTransformer.MODE_IQT)
        #if torch.cuda.is_available():
        #    model.cuda()
        model.to(self.device)
        opt = torch.optim.Adam(lr=cmd_args.lr, params=model.parameters())
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (cmd_args.lr_warmup / cmd_args.batch_size), 1.0))
        # training loop
        seen = 0
        for e in range(cmd_args.num_epochs):
            print(f'\n epoch {e}')
            model.train(True)
            # M3
            #for batch in tqdm.tqdm(train_iter):
            for X, y in tqdm.tqdm(train_iter):
                opt.zero_grad()
                # M4
                X = X.float().to(self.device)
                y = y.long().to(self.device)
                #X = batch.text[0]
                #y = batch.label - 1
                if X.size(1) > mx:
                    X = X[:, :mx]
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
                # M5              
                for X, y in tqdm.tqdm(test_iter):
                    X = X.float().to(self.device)
                    y = y.long().to(self.device)
                #for batch in test_iter:
                #    X = batch.text[0]
                #    Y = batch.label - 1
                    if X.size(1) > mx:
                        X = X[:, :mx]
                    y_hat = model(X).argmax(dim=1)
                    tot += float(X.size(0))
                    cor += float((y == y_hat).sum().item())
                acc = cor / tot
                print(f'-- {"test" if cmd_args.final else "validation"} accuracy {acc:.3}')








    def parse_args(self):
        parser = ArgumentParser()
        parser.add_argument('--mode', dest='run_mode', default='bizdmrl', type=str)
        parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)
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

        

    def load_dataset(self, ds_mode):
        if IqttApp.DSM_IMDB == ds_mode:
            return self.load_imdb_dataset()
        elif IqttApp.DSM_STOCK == ds_mode:
            return self.load_stock_dataset()
        else:
            print('未知数据源')
            exit(0)

    def load_imdb_dataset(self, cmd_args):
        # Used for converting between nats and bits
        LOG2E = math.log2(math.e)
        TEXT = torchtext.legacy.data.Field(lower=True, include_lengths=True, batch_first=True)
        LABEL = torchtext.legacy.data.Field(sequential=False)
        NUM_CLS = 2
        # load the IMDB data
        if cmd_args.final:
            train, test = datasets.IMDB.splits(TEXT, LABEL)
            TEXT.build_vocab(train, max_size=cmd_args.vocab_size - 2)
            LABEL.build_vocab(train)
            train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=cmd_args.batch_size, device=IqttUtil.d())
        else:
            tdata, _ = datasets.IMDB.splits(TEXT, LABEL)
            train, test = tdata.split(split_ratio=0.8)
            TEXT.build_vocab(train, max_size=cmd_args.vocab_size - 2) # - 2 to make space for <unk> and <pad>
            LABEL.build_vocab(train)
            train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=cmd_args.batch_size, device=IqttUtil.d())
        print(f'- nr. of training examples {len(train_iter)}')
        print(f'- nr. of {"test" if cmd_args.final else "validation"} examples {len(test_iter)}')
        if cmd_args.max_length < 0:
            mx = max([input.text[0].size(1) for input in train_iter])
            mx = mx * 2
            print(f'- maximum sequence length: {mx}')
        else:
            mx = cmd_args.max_length
        seq_length = mx
        return train_iter, test_iter, NUM_CLS, seq_length, mx

    def load_stock_dataset(self, cmd_args):
        stock_symbol = 'sh600260'
        train_ds = AksDs(stock_symbol, \
                    ds_mode=AksDs.DS_MODE_TRAIN, train_rate=0.1, val_rate=0.0, test_rate=0.02)
        train_loader = DataLoader(
            train_ds,
            batch_size = cmd_args.batch_size,
            num_workers = 0,
            shuffle = True,
            drop_last = True
        )
        train_iter = train_loader
        #test_ds = AksDs(stock_symbol, \
        #            ds_mode=AksDs.DS_MODE_TEST, train_rate=0.98, val_rate=0.0, test_rate=0.02)
        test_ds = AksDs(stock_symbol, \
                    ds_mode=AksDs.DS_MODE_TRAIN, train_rate=0.1, val_rate=0.0, test_rate=0.02)
        test_loader = DataLoader(
            test_ds,
            batch_size = cmd_args.batch_size,
            num_workers = 0,
            shuffle = True,
            drop_last = True
        )
        test_iter = test_loader
        NUM_CLS = 3
        cmd_args.embedding_size = 5
        seq_length = 10
        cmd_args.num_heads = 4
        cmd_args.depth = 2
        mx = cmd_args.embedding_size
        return train_iter, test_iter, NUM_CLS, seq_length, mx












    def t1(self):
        sa = SelfAttention(emb=5, heads=2, mask=False)
        x = torch.rand(8, 12, 5)
        y = sa(x)
        print('x: {0}; y: {1};'.format(x.shape, y.shape))