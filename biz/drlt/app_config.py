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
        'GAMMA': 0.99,
        'LEARNING_RATE': 0.001,
        'ENTROPY_BETA': 0.01,
        'REWARD_STEPS': 4,
        'CLIP_GRAD': 0.1,
        'PROCESSES_COUNT': 2, # 4
        'NUM_ENVS': 4, #8
        'GRAD_BATCH': 32, #64
        'TRAIN_BATCH': 2,
        'ENV_NAME': "PongNoFrameskip-v4",
        'NAME': 'pong',
        'REWARD_BOUND': 18
    }
    