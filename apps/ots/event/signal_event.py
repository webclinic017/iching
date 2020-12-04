#
from apps.ots.event.ots_event import OtsEvent

class SignalEvent(OtsEvent):
    ''' 
    处理从Strategy对象发过来的信号事件，信号会被 Portfolio对象
    接收并且根据这个信号来采取行动 
    '''
    def __init__(self, strategy_id, symbol, occur_time, direction, strength):
        self.strategy = strategy_id
        self.type = OtsEvent.ET_SIGNAL
        self.symbol = symbol
        self.occur_time = occur_time
        self.direction = direction # 是买还是卖
        self.strength = strength