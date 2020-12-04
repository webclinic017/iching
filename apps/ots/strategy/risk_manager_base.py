#
from apps.ots.event.signal_event import SignalEvent

class RiskManagerBase(object):
    def __init__(self):
        self.refl = ''

    def get_mkt_quantity(self, signalEvent):
        return 100