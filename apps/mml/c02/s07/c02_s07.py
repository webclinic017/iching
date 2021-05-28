#
from apps.mml.c02.s07.c02_s07_e01 import C02S07E01

class C02S07(object):
    def __init__(self):
        self.name = 'apps.mml.c02.s07.c02_s07.C02S07'

    def startup(self, args={}):
        print('2.7 线性变换')
        exp = C02S07E01()
        exp.startup()