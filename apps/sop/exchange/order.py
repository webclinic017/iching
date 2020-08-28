# 订单类

class Order(object):
    def __init__(self, action):
        self.refl = 'apps.sop.exchange.Order'

    def __str__(self):
        msg = '订单类：'
        return msg