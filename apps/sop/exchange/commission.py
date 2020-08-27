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

    def calculate_security_deposit(self, option_contract, asset_price, quant):
        '''
        计算期权合约卖方保证金金额
        参数：
            option_type 有认购期权和认沽期权
            option_price 单位期权合约价格
            exercise_price 行权价格
            asset_price 标的价格
            quant 数量
        '''
        return option_contract.calculate_security_deposit(asset_price) * quant