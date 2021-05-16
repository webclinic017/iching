#
from apps.drl.c02.c02_e01 import C02E01

class C02App(object):
    def __init__(self):
        self.name = 'apps.drl.c02.c02_app.C02App'

    def startup(self, args={}):
        exp = C02E01()
        exp.startup()