#
import numpy as np
import unittest
from biz.drlt.app_config import AppConfig
from biz.drlt.rll.actions import EpsilonGreedyActionSelector

class TEpsilonGreedyActionSelector(unittest.TestCase):
    def test_usage(self):
        selector = EpsilonGreedyActionSelector(AppConfig.EPS_START)
        self.assertTrue(1>0)
        scores = np.array([
            [0.3, 0.2, 0.5, 0.1, 0.4],
            [0.11, 0.52, 0.33, 0.65, 0.27],
            [0.98, 0.32, 0.99, 0.15, 0.57]
        ])
        action = selector(scores)
        print('action: {0};'.format(action))