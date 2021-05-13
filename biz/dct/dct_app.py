#
from biz.dct.dct_ds import DctDs

class DctApp(object):
    def __init__(self):
        self.name = 'apps.dct.dct_app.DctApp'

    def startup(self, args={}):
        print('数字货币交易系统')
        ds = DctDs()
        ds.load_data()