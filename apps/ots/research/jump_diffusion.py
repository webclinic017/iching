#
import numpy as np
from apps.ots.nc_util import NcUtil
from apps.ots.simulation_base import SimulationBase

class JumpDiffusion(SimulationBase):
    ''' 基于merton 1976 jump diffusion模型生成模拟路径 '''
    def __init__(self, name, mkt_env, corr=False):
        super(JumpDiffusion, self).__init__(name, mkt_env, corr)
        self.lamb = mkt_env.get_const('lambda')
        self.mu = mkt_env.get_const('mu')
        self.delta = mkt_env.get_const('delta')

    def update(self, initial_value=None, volatility=None, lamb=None, mu=None, delta=None, final_date=None):
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if lamb is not None:
            self.lamb = lamb
        if mu is not None:
            self.mu = mu
        if delta is not None:
            self.delta = delta
        if final_date is not None:
            self.final_date = final_date
        self.instrument_values = None

    def generate_paths(self, fixed_seed=False, day_count=365.0):
        if self.time_grid is None:
            self.generate_time_grid()
        M = len(self.time_grid)
        I = self.paths
        paths = np.zeros((M, I))
        paths[0] = self.initial_value
        if self.correlated is False:
            sn1 = NcUtil.gen_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            sn1 = self.random_numbers
        sn2 = NcUtil.gen_random_numbers((1, M, I), fixed_seed=fixed_seed)
        rj = self.lamb * (np.exp(self.mu + 0.5*self.delta**2) - 1)
        short_rate = self.discount_curve.short_rate
        for t in range(1, len(self.time_grid)):
            if self.correlated is False:
                ran = sn1[t]
            else:
                ran = np.dot(self.cholesky_matrix, sn1[:, t, :])
                ran = ran[self.rn_set]
            dt = (self.time_grid[t] - self.time_grid[t-1]).days / day_count
            poi = np.random.poisson(self.lamb*dt, I)
            paths[t] = paths[t-1]*(np.exp((short_rate - rj - 
                            0.5*self.volatility**2)*dt + 
                            self.volatility*np.sqrt(dt)*ran) 
                            +(np.exp(self.mu + self.delta*sn2[t])-1)*poi)
        self.instrument_values = paths
        