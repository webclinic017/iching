#
from apps.mml.c02.c02_app import C02App

class MmlApp(object):
    def __init__(self):
        self.name = 'apps.mml.mml_app.MmlApp'

    def startup(self, args={}):
        print('机器学习中的数学')
        app = C02App()
        app.startup()