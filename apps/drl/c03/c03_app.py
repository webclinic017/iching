#
from apps.drl.c03.c03_e01 import C03E01
from apps.drl.c03.e02.c03_e02 import C03E02
from apps.drl.c03.e03.c03_e03 import C03E03

class C03App(object):
    def __init__(self):
        self.name = 'apps.drl.c03.c03_app.C03App'

    def startup(self, args={}):
        #exp = C03E01()
        #exp = C03E02()
        exp = C03E03()
        exp.startup()