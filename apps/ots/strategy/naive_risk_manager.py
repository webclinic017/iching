#
from apps.ots.event.signal_event import SignalEvent
from apps.ots.strategy.risk_manager_base import RiskManagerBase

class NaiveRiskManager(RiskManagerBase):
    def __init__(self):
        self.refl = ''

    def get_mkt_quantity(self, signalEvent):
        return 100