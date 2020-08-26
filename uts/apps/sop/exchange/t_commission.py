# 交易费用计算类Commission的测试类
import unittest
from apps.sop.exchange.commission import Commission

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