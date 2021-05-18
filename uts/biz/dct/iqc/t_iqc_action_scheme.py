#
import unittest
import numpy as np
from iqt.oms.instruments.instrument import Instrument
from iqt.oms.exchanges import Exchange
from iqt.oms.services.execution.simulated import execute_order
from iqt.feed.core import Stream
from biz.dct.iqc.iqc_action_scheme import IqcActionScheme
from iqt.oms.wallets import Wallet

class TIqcActionScheme(unittest.TestCase):
    def test_action_space(self):
        CNY = Instrument("CNY", 2, "China RMB")
        IQC = Instrument("IQC", 8, "Iching Quantitative Coin")

        x = np.arange(0, 2*np.pi, 2*np.pi / 1001)
        y = 50*np.sin(3*x) + 100

        x = np.arange(0, 2*np.pi, 2*np.pi / 1000)
        iqc_stream = Stream.source(y, dtype="float").rename("CNY-IQC")

        iqcex = Exchange("iqcex", service=execute_order)(
            iqc_stream
        )
        cash = Wallet(iqcex, 100000 * CNY)
        asset = Wallet(iqcex, 0*IQC)
        ias = IqcActionScheme(cash=cash, asset=asset)
        print('action_space: {0};'.format(ias.action_space.sample()))
        print('action_space: {0};'.format(ias.action_space.sample()))
        print('action_space: {0};'.format(ias.action_space.sample()))
        print('action_space: {0};'.format(ias.action_space.sample()))
        print('action_space: {0};'.format(ias.action_space.sample()))
        print('action_space: {0};'.format(ias.action_space.sample()))