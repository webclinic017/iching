#
from apps.drl.chp001.exp001_001 import Exp001001
from apps.drl.chp001.exp001_002 import Exp001002

class DrlApp(object):
    def __init__(self):
        self.refl = 'apps.drl.DrlApp'

    def startup(self):
        #exp = Exp001001()
        exp = Exp001002()
        exp.startup()