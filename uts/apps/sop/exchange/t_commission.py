# 交易费用计算类Commission的测试类
import unittest
from apps.sop.exchange.commission import Commission
from apps.sop.exchange.option_contract import OptionContract

class TCommission(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def test_calculate_royalty(self):
        cmn = Commission()
        royalty = cmn.calculate_royalty(0.1, 28)
        self.assertTrue(abs(royalty - 28000))

    def test_calculate_security_deposit(self):
        comm = Commission()
        oc = OptionContract()
        oc.exercise_price = 22.0
        oc.price = 0.40
        oc.option_contract_type = OptionContract.OCT_CALL
        oc.side = OptionContract.SIDE_SHORT
        asset_price = 21.5
        quant = 10
        amount = comm.calculate_security_deposit(oc, asset_price, quant)
        print(amount)