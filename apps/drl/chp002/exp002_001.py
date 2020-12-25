#
import numpy as np
import gym
import gym_bandits

class Exp002001(object):
    def __init__(self):
        self.refl = 'apps.drl.chp002.Exp002001'

    def startup(self):
        SEEDS = (12, 34, 56, 78, 90)
        b2_Vs = []
        for seed in SEEDS:
            env_name = 'BanditTwoArmedUniform-v0'
            #env_name = 'BanditTwoArmedDeterministicFixed-v0'
            env = gym.make(env_name, seed=seed) ; env.reset()
            b2_Q = np.array(env.env.p_dist * env.env.r_dist)
            print('Two-Armed Bandit environment with seed', seed)
            print('Probability of reward:', env.env.p_dist)
            print('Reward:', env.env.r_dist)
            print('Q(.):', b2_Q)
            b2_Vs.append(np.max(b2_Q))
            print('V*:', b2_Vs[-1])
            print()
        print('Mean V* across all seeds:', np.mean(b2_Vs))