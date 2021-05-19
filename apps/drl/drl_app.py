#
from apps.drl.chp001.exp001_001 import Exp001001
from apps.drl.chp001.exp001_002 import Exp001002
from apps.drl.chp002.exp002_001 import Exp002001
from apps.drl.chpA01.chp_a01_main import ChpA01Main
from apps.drl.chpA01.e02.chp_a01_e02_main import ChpA01E02Main
from apps.drl.c02.c02_app import C02App
from apps.drl.c03.c03_app import C03App
from apps.drl.c04.c04_app import C04App

class DrlApp(object):
    def __init__(self):
        self.refl = 'apps.drl.DrlApp'

    def startup(self):
        #exp = Exp001001()
        #exp = Exp001002()
        #exp = Exp002001()
        #exp = ChpA01Main()
        #exp = ChpA01Main()

        #app = C02App()
        #app = C03App()
        app = C04App()
        app.startup()