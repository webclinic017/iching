#
from __future__ import print_function
import datetime as dt
from apps.ots.broker.order_executor import OrderExecutor
from apps.ots.event.ots_event import OtsEvent
from apps.ots.event.order_event import OrderEvent
from apps.ots.event.fill_event import FillEvent

class SimulatedOrderExecutor(OrderExecutor):
    '''
    模拟订单执行，简单执行整个订单，不考虑滑价，市价成交时的延时，订单量大时分批成交的情况
    '''
    def __init__(self, events):
        self.events = events

    def execute_order(self, order_event):
        '''
        将订单事件转化为FillEvent
        '''
        if order_event.type == OtsEvent.ET_ORDER:
            fill_event = FillEvent(
                dt.datetime.utcnow(),
                order_event.symbol,
                'EXCHANGE_ICHING', order_event.quantity,
                order_event.direction, None
            )
            self.events.put(fill_event)
