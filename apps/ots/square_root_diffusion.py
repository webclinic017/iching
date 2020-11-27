#
import numpy as np
from apps.ots.nc_util import NcUtil
from apps.ots.simulation_base import SimulationBase

class SquareRootDiffusion(SimulationBase):
    ''' 生成基于Cox-Ingersoll-Ross(1985)平方根扩散模型生成数据 '''
    def __init__(self, name, mkt_env, corr=False):
        super(SquareRootDiffusion, self).__init__(name, mkt_env, corr)
        self.kappa = mkt_env.get_const('kappa')
        self.theta = mkt_env.get_const('theta')

    def update(self, initial_value=None, volatility=None, kappa=None, theta=None, final_date=None):
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if kappa is not None:
            self.kappa = kappa
        if theta is not None:
            self.theta = theta
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=True, day_count=365.0):
        if self.time_grid is None:
            self.generate_time_grid()
        M = len(self.time_grid)
        I = self.paths
        paths = np.zeros((M, I))
        paths_ = np.zeros_like(paths)
        paths[0] = self.initial_value
        paths_[0] = self.initial_value
        if self.correlated is False:
            rand = NcUtil.gen_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            rand = self.random_numbers
        for t in range(1, len(self.time_grid)):
            dt = (self.time_grid[t] - self.time_grid[t-1]).days / day_count
            if self.correlated is False:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]
            paths_[t] = (paths_[t-1] + self.kappa
                        *(self.theta - np.maximum(0, paths_[t-1, :])) * dt
                        + np.sqrt(np.maximum(0, paths_[t-1, :]))
                        * self.volatility * np.sqrt(dt)*ran)
            paths[t] = np.maximum(0, paths_[t])
        self.instrument_values = paths