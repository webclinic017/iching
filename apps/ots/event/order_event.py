#
from apps.ots.event.ots_event import OtsEvent

class OrderEvent(OtsEvent):
    '''
    处理向执行系统提交的订单信息，包括：symbol、类型（市价 or 现价）、数量、方向
    '''
    def __init__(self, symbol, order_type,  quantity, direction):
        self.type = OtsEvent.ET_ORDER
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction

    def print_order(self):
        print('订单：代码：{0}；类型：{1}；数量：{2}；方向：{3}；'.format(
                    self.symbol, self.order_type, self.quantity, self.direction))