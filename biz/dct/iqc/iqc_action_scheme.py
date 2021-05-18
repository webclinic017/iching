#
from gym.spaces import Discrete

from iqt.env.default.actions import IqtActionScheme

from iqt.env.generic import ActionScheme, TradingEnv
from iqt.core import Clock
from iqt.oms.instruments import ExchangePair
from iqt.oms.wallets import Portfolio
from iqt.oms.orders import (
    Order,
    proportion_order,
    TradeSide,
    TradeType
)

class IqcActionScheme(IqtActionScheme):
    registered_name = 'iqc_action_scheme'

    def __init__(self, cash: 'Wallet', asset: 'Wallet'):
        super().__init__()
        self.cash = cash
        self.asset = asset
        self.listeners = []
        self.action = 0

    @property
    def action_space(self):
        return Discrete(2)

    def get_orders(self, action: int, portfolio: 'Portfolio'):
        pass

    def attach(self, listener):
        pass

    def reset(self):
        super().reset()
        action = 0