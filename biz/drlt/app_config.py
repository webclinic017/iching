#
import pathlib

class AppConfig(object):
    SAVES_DIR = pathlib.Path("./work/saves")
    STOCKS = "data/YNDX_160101_161231.csv"
    VAL_STOCKS = "data/YNDX_150101_151231.csv"

    BATCH_SIZE = 32
    BARS_COUNT = 10

    EPS_START = 1.0
    EPS_FINAL = 0.1
    EPS_STEPS = 1000000

    GAMMA = 0.99

    REPLAY_SIZE = 100000
    REPLAY_INITIAL = 10000
    REWARD_STEPS = 2
    LEARNING_RATE = 0.0001
    STATES_TO_EVALUATE = 1000

    DEFAULT_BARS_COUNT = 10
    DEFAULT_COMMISSION_PERC = 0.1

    # A3C
    a3c_config = {
        'gamma': 0.99,
        'learning_rate': 0.001,
        'entropy_beta': 0.01,
        'reward_steps': 4,
        'clip_grad': 0.1,
        'processes_count': 2, # 4
        'num_envs': 4, #8
        'grad_batch': 32, #64
        'train_batch': 2,
        'env_name': "stock-minute-bar-v1",
        'name': 'minuteBar',
        'reward_bound': 18
    }
    