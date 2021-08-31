# 金融市场交易系统（Financial Market Trading System）
from argparse import ArgumentParser
from apps.fmts.conf.app_config import AppConfig
from apps.fmts.ds.ohlcv_dataset import OhlcvDataset

class FmtsApp(object):
    def __init__(self):
        self.name = 'apps.fmts.fmts_app.FmtsApp'

    def startup(self, args={}):
        print('金融市场交易系统 v0.0.3')
        cmd_args = self.parse_args()

    def load_stock_dataset(self, cmd_args):
        stock_symbol = 'sh600260'
        '''
        train_ds = OhlcvDataset(stock_symbol, \
                    ds_mode=OhlcvDataset.DS_MODE_TRAIN, train_rate=0.1, val_rate=0.0, test_rate=0.02)
        train_loader = DataLoader(
            train_ds,
            batch_size = cmd_args.batch_size,
            num_workers = 0,
            shuffle = True,
            drop_last = True
        )
        train_iter = train_loader
        test_ds = OhlcvDataset(stock_symbol, \
                    ds_mode=OhlcvDataset.DS_MODE_TRAIN, train_rate=0.1, val_rate=0.0, test_rate=0.02)
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
        '''

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
