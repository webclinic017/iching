#
from apps.drl.c04.e01.c04_e01 import C04E01

class C04App(object):
    def __init__(self):
        self.name = 'apps.drl.c04.c04_app.C04App'

    def startup(self, args={}):
        exp = C04E01()
        exp.startup()