# 
from apps.ots.event.ots_event import OtsEvent

class MarketEvent(OtsEvent):
    ''' 处理接收到新的市场数据的更新 '''
    def __init__(self):
        self.type = OtsEvent.ET_MARKET