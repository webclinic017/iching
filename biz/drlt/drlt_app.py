# Deep Reinforcement Learning Trader v0.0.1
from biz.drlt.dqn_app import DqnApp
from biz.drlt.a3c_app import A3cApp

class DrltApp(object):
    def __init__(self):
        self.name = 'biz.drlt.drlt_app.DrltApp'

    def startup(self, args={}):
        print('深度强化学习交易系统 v0.0.2')
        # app = DqnApp()
        app = A3cApp()
        app.train()