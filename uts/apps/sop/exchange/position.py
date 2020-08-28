# 仓位管理类

class Position(object):
    def __init__(self):
        self.name = 'apps.sop.exchange.Position'
        self.position_id = 1 # 
        self.amount = 0.0 # 现金账户
        self.net_worth = 0.0 # 净值
        self.call_options = [] # 持有的认购合约列表
        self.put_options = [] # 持有的认沽合约列表
        self.rpnl = 0.0 # 已实现损益
        self.upnl = 0.0 # 未实现损益