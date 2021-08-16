# 金融市场交易系统（Financial Market Trading System）
from apps.fmts.conf.app_config import AppConfig

class FmtsApp(object):
    def __init__(self):
        self.name = 'apps.fmts.fmts_app.FmtsApp'

    def startup(self, args={}):
        print('金融市场交易系统 v0.0.1')
        print('run_mode={0}; num_epoch={1};'.format(args['run_mode'], args['num_epochs']))
        print('config version: {0};'.format(AppConfig.version))
