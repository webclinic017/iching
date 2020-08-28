# 券商类

class Broker(object):
    def __init__(self):
        self.refl = 'apps.sop.exchange.Broker'

    def execute_order(self, order):
        print('券商系统订单执行完毕')