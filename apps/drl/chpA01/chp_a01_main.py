#
#from apps.drl.chpA01.e01.chp_a01_e01 import ChpA01E01
#from apps.drl.chpA01.e02.chp_a01_e02_main import ChpA01E02Main
from apps.drl.chpA01.e03.chp_a01_e03_main import ChpA01E03Main

class ChpA01Main(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print('chpA01 e01')
        exp = ChpA01E03Main()
        exp.startup()