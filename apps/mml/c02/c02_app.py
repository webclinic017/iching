# 
from apps.mml.c02.c02_e01 import C02E01
from apps.mml.c02.c02_e02 import C02E02
from apps.mml.c02.s07.c02_s07 import C02S07

class C02App(object):
    def __init__(self):
        self.name = 'apps.mml.c02.c02_app.C02App'

    def startup(self):
        print('第二章 线性代数')
        #exp = C02E01()
        #exp = C02E02()
        exp = C02S07()
        exp.startup()