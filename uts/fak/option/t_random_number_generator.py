#
import unittest
from fak.option.random_number_generator import RandomNumberGenerator

class TRandomNumberGenerator(unittest.TestCase):
    def test_sn_random_numbers1(self):
        rng = RandomNumberGenerator.sn_random_numbers((2, 2, 2), antithetic=False, moment_matching=False, fixed_seed=True)
        print('random numbers: {0}; mu={1}, std={2};'.format(
            rng,
            round(rng.mean(), 6),
            round(rng.std(), 6)
        ))
        
    def test_sn_random_numbers2(self):
        rng = RandomNumberGenerator.sn_random_numbers((2, 2, 2), antithetic=False, moment_matching=True, fixed_seed=True)
        print('random numbers: {0}; mu={1}, std={2};'.format(
            rng,
            round(rng.mean(), 6),
            round(rng.std(), 6)
        ))