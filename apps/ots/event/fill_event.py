#
from apps.ots.event.ots_event import OtsEvent

class FillEvent(OtsEvent):
    '''
    负责订单执行，由交易所返回，包括交易信息：执行的数量、价格、交易佣金和手续费，
    目前不支持一个订单有多个价格（以后可以扩充）
    '''
    def __init__(self, timeindex, symbol, exchange, quantity, direction, fill_cost, commission=None):
        self.type = OtsEvent.ET_FILL
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        if commission is None:
            self.commission = self.calculate_ib_commission()
        else:
            self.commission = commission

    def calculate_ib_commission(self):
        ''' 计算基于interactive broker的交易费用 '''
        full_cost = 1.3
        if self.quantity <= 500:
            full_cost = max(1.3, 0.013*self.quantity)
        else:
            full_cost = max(1.3, 0.008*self.quantity)
        return full_cost
