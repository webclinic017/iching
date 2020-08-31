# 系统配置信息

class SopConfig(object):
    adjust_rate = 0.1
    min_adjust_rate = 0.05
    contract_unit = 10000
    exercise_price_delta = 2.0
    lookback_num = 5 # 向前看几个时间点（tick）
    # 以元为单位金额后面的乘数，用于保证金额精度
    cash_sacle = 10000
    dq_size = 373 # 每日所有期权合约和标的物行情大小
    docq_size = 8 # 每日单个期权合约大小
    duaq_size = 5 # 每日标的物行情大小