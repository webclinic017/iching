#
from __future__ import print_function
from abc import ABCMeta, abstractmethod
import datetime as dt
try:
    import Queue as queque
except ImportError:
    import queue
from apps.ots.event.ots_event import OtsEvent
from apps.ots.event.order_event import OrderEvent
from apps.ots.event.fill_event import FillEvent

class OrderExecutor(object):
    ''' 
    处理由Portfolio生成的OrderEvent，与实际市场中发生的FillEvent，
    既可以实盘也可以是回测过程
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def execute_order(self, order_event):
        ''' 执行OrderEvent对象，产生FillEvent并放入到事件队列中 '''
        raise NotImplementedError('必须实现execute_order方法')