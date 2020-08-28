# 仓位管理类
from apps.sop.sop_config import SopConfig

class Position(object):
    ID_BASE = 1

    def __init__(self, amount):
        self.name = 'apps.sop.exchange.Position'
        self.position_id = Position.ID_BASE # 
        Position.ID_BASE += 1
        self.amount = amount * SopConfig.cash_sacle # 现金账户 元 * 1000000
        self.net_worth = 0.0 # 净值
        self.call_options = [] # 持有的认购合约列表
        self.put_options = [] # 持有的认沽合约列表
        self.rpnl = 0.0 # 已实现损益
        self.upnl = 0.0 # 未实现损益

    def __str__(self):
        msg = '仓位信息：\n'
        msg += '    编号：{0};\n'.format(self.position_id)
        msg += '    金额：{0:0.2f};\n'.format(self.amount / SopConfig.cash_sacle)
        return msg