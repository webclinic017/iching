#
import collections

class AppConfig(object):
    DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
    MEAN_REWARD_BOUND = 19

    GAMMA = 0.99
    BATCH_SIZE = 32
    REPLAY_SIZE = 10000
    LEARNING_RATE = 1e-4
    SYNC_TARGET_FRAMES = 1000
    REPLAY_START_SIZE = 10000

    EPSILON_DECAY_LAST_FRAME = 150000
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.01

    FPS = 25


    Experience = collections.namedtuple(
        'Experience', field_names=['state', 'action', 'reward',
                                'done', 'new_state'])