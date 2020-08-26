# 交易费用相关的计算类
from apps.sop.sop_config import SopConfig

class Commission(object):
    def __init__(self):
        self.refl = 'apps.sop.exchange.Commission'

    def calculate_royalty(self, price, quant):
        '''
        计算期权交易的权利金，权利金直接由买方转给卖方
        参数：
            price 合约价格
            quant 多少手，1手为10000份
        '''
        return price * quant * SopConfig.contract_unit