#

class AppConfig(object):
    # 定义当前市场所处的行情
    MR_VIBRATE = 0 # 震荡行情
    MR_BULL = 1 # 上涨
    MR_BEAR = 2 # 下跌

    # 决定市场涨跌的参数
    mr_params = {
        'asc_span_coff': 0.5, # std的变化系数，越小则表明越容易出现上涨或下跌模式
        'desc_span_coff': 0.3,
        'asc_threshold': 0.008,
        'desc_threshold': 0.006
    }

    # 历史行情状态长度，未来市场变化长度
    mdp_params = {
        'back_window': 10, # 向前10分钟作为一帧样本
        'forward_step': 60 # 向前看1小时来判断市场状态
    }