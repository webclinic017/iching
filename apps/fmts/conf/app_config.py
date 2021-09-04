# 系统配置类

class AppConfig(object):
    version = 'v0.0.1'
    # Transformer相关配置信息
    fmts_transformer = {
        'app_mode_imdb': 1,
        'app_mode_iqt': 2,
        # 
        'task_mode_random': 1,
        'task_mode_ts': 2 # 时序信号
    }
    