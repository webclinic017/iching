#
import gym, gym_walk
import numpy as np
from pprint import pprint
from tqdm import tqdm_notebook as tqdm

from itertools import cycle

import random

class Exp001002(object):
    def __init__(self):
        self.refl = 'apps.drl.chp001.Exp001002'

    def startup(self):
        print('MDP应用')
        # 初始化环境
        env = gym.make('SlipperyWalkFive-v0')
        P = env.env.P
        init_state = env.reset()
        goal_state = 6
        LEFT, RIGHT = range(2)
        pi = lambda s: {
            0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
        }[s]
        #self.print_policy(pi, P, action_symbols=('<', '>'), n_cols=7)
        state = 2
        self.test001(P, state)

    def print_policy(self, pi, P, action_symbols=('<', 'v', '>', '^'), 
                n_cols=4, title='Policy:'):
        print(title)
        arrs = {k:v for k,v in enumerate(action_symbols)}
        for s in range(len(P)):
            a = pi(s)
            print("| ", end="")
            if np.all([done for action in P[s].values() 
                        for _, _, _, done in action]):
                print("".rjust(9), end=" ")
            else:
                print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
            if (s + 1) % n_cols == 0: print("|")

    def test001(self, P, state):
        v1 = [action for action in P[state].values()]
        print('v1: {0};'.format(v1))
        v2 = [done for action in P[state].values() for _, _, _, done in action]
        print('v2: {0};'.format(v2))
        v3 = np.all(v2)
        print('v3: {0};'.format(v3))