#

class AppConfig(object):
    GAMMA = 0.99
    LEARNING_RATE = 0.001
    ENTROPY_BETA = 0.01

    REWARD_STEPS = 4
    CLIP_GRAD = 0.1

    PROCESSES_COUNT = 4
    NUM_ENVS = 8

    GRAD_BATCH = 64
    TRAIN_BATCH = 2

    ENV_NAME = "PongNoFrameskip-v4"
    NAME = 'pong'
    REWARD_BOUND = 18