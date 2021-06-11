# Epsilon greedy算法中epsilon值逐渐衰减策略
import unittest
from biz.drlt.rll.actions import EpsilonGreedyActionSelector
from biz.drlt.rll.actions import EpsilonTracker

class TEpsilonTracker(unittest.TestCase):
    def test_exp(self):
        selector = EpsilonGreedyActionSelector()
        et = EpsilonTracker(selector=selector, eps_start=1.0, eps_final=0.05, eps_frames=100)
        for i in range(100):
            et.frame(i)
            print('{0}: epsilon={1};'.format(i, et.selector.epsilon))