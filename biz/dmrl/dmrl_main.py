#
from biz.dmrl.maml_app import MamlApp

class DmrlMain(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print("元强化学习量化交易系统 v0.0.1")
        app = MamlApp()
        app.startup()