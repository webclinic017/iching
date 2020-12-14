#
from apps.drl.chp001.exp001_001 import Exp001001

class DrlApp(object):
    def __init__(self):
        self.refl = 'apps.drl.DrlApp'

    def startup(self):
        exp = Exp001001()
        exp.startup()