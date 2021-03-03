#
import gym #, gym_walk
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
        self.print_policy(pi, P, action_symbols=('<', '>'), n_cols=7)
        prob = self.probability_success(env, pi, goal_state)
        print('获胜概率：{0};'.format(prob))
        g_mean = self.mean_return(env, pi)
        print('平均回报：{0};'.format(g_mean))
        V = self.policy_evaluation(pi, P)
        self.print_state_value_function(V, P, n_cols=7, prec=5)
        improved_pi = self.policy_improvement(V, P)
        self.print_policy(improved_pi, P, action_symbols=('<', '>'), n_cols=7)
        # policy iteration
        print('PI: Policy Iteration')
        optimal_V, optimal_pi = self.policy_iteration(P)
        self.print_policy(optimal_pi, P, action_symbols=('<', '>'), n_cols=7)
        self.print_state_value_function(optimal_V, P, n_cols=7, prec=5)
        print('VI: Value Iteration')
        V2, pi2 = self.value_iteration(P)
        self.print_policy(optimal_pi, P, action_symbols=('<', '>'), n_cols=7)
        self.print_state_value_function(optimal_V, P, n_cols=7, prec=5)
        

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

    def probability_success(self, env, pi, goal_state, n_episodes=100, max_steps=200):
        '''
        '''
        random.seed(123); np.random.seed(123) ; env.seed(123)
        results = []
        for epoch in range(n_episodes):
            state, done, steps = env.reset(), False, 0
            while not done and steps < max_steps:
                state, _, done, h = env.step(pi(state))
                steps += 1
            results.append(state == goal_state)
        return np.sum(results)/len(results)

    def mean_return(self, env, pi, n_episodes=100, max_steps=200):
        random.seed(123); np.random.seed(123) ; env.seed(123)
        results = []
        for _ in range(n_episodes):
            state, done, steps = env.reset(), False, 0
            results.append(0.0)
            while not done and steps < max_steps:
                state, reward, done, _ = env.step(pi(state))
                results[-1] += reward
                steps += 1
        return np.mean(results)

    def policy_evaluation(self, pi, P, gamma=1.0, theta=1e-10):
        prev_V = np.zeros(len(P), dtype=np.float64)
        while True:
            V = np.zeros(len(P), dtype=np.float64)
            for s in range(len(P)):
                for prob, next_state, reward, done in P[s][pi(s)]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
            if np.max(np.abs(prev_V - V)) < theta:
                break
            prev_V = V.copy()
        return V

    def print_state_value_function(self, V, P, n_cols=4, prec=3, title='State-value function:'):
        print(title)
        for s in range(len(P)):
            v = V[s]
            print("| ", end="")
            if np.all([done for action in P[s].values() for _, _, _, done in action]):
                print("".rjust(9), end=" ")
            else:
                print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
            if (s + 1) % n_cols == 0: print("|")
    
    def policy_improvement(self, V, P, gamma=1.0):
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + 
                            gamma * V[next_state] * (not done))
        new_pi = lambda s: {s:a for s, a 
            in enumerate(np.argmax(Q, axis=1))
        }[s]
        return new_pi

    def policy_iteration(self, P, gamma=1.0, theta=1e-10):
        random_actions = np.random.choice(tuple(P[0].keys()), len(P))
        pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
        while True:
            old_pi = {s:pi(s) for s in range(len(P))}
            V = self.policy_evaluation(pi, P, gamma, theta)
            pi = self.policy_improvement(V, P, gamma)
            if old_pi == {s:pi(s) for s in range(len(P))}:
                break
        return V, pi

    def value_iteration(self, P, gamma=1.0, theta=1e-10):
        V = np.zeros(len(P), dtype=np.float64)
        while True:
            Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
            for s in range(len(P)):
                for a in range(len(P[s])):
                    for prob, next_state, reward, done in P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
                break
            V = np.max(Q, axis=1)
        pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
        return V, pi












    def test001(self, P, state):
        v1 = [action for action in P[state].values()]
        print('v1: {0};'.format(v1))
        v2 = [done for action in P[state].values() for _, _, _, done in action]
        print('v2: {0};'.format(v2))
        v3 = np.all(v2)
        print('v3: {0};'.format(v3))