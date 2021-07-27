#
from biz.dmrl.maml_app import MamlApp
from biz.dmrl.aks_env import AksEnv
from biz.dmrl.aks_util import AksUtil

class DmrlMain(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print("元强化学习量化交易系统 v0.0.1")
        #app = MamlApp()
        #app.startup()
        # (self, stock_symbol, n_way, k_shot, q_query, ds_mode=0, train_rate=0.0, val_rate=0.0, test_rate=0.0):
        stock_symbol = 'sh600260'
        env = AksEnv(stock_symbol)
        env.reset()