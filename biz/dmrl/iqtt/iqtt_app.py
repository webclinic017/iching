#

class IqttApp(object):
    def __init__(self):
        self.name = 'biz.dmrl.iqtt.iqtt_app.IqttApp'

    def startup(self, args={}):
        print('Iching Quantitative Trading Transformer v0.0.1')