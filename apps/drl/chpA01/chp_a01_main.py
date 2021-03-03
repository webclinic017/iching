#
from apps.drl.chpA01.e01.chp_a01_e01 import ChpA01E01

class ChpA01Main(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print('chpA01 e01')
        exp = ChpA01E01()
        exp.startup()