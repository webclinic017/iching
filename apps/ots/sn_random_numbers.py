#
import numpy as np

def sn_random_numbers(shape, antithetic=True, moment_matching=True, fixed_seed=True):
    if fixed_seed:
        np.random.seed(1000)
    if antithetic:
        ran = np.random.standard_normal((shape[0], shape[1], shape[2]/2))
        ran = np.concatenate((ran, ran), axis=2)
    else:
        ran = np.random.stand_normal(shape)
    if moment_matching:
        ran = ran - np.mean(ran)
        ran = ran / np.std(ran)
    if shape[0] == 1:
        return ran[0]
    else:
        return ran