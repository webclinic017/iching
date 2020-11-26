#
import numpy as np
import pandas as pd

class SimulationBase(object):
    ''' 模拟基类 '''
    def __init__(self, name, mkt_env, corr):
        self.name = name
        self.pricing_date = mkt_env.pricing_date
        self.initial_value = mkt_env.get_const('initial_value')
        self.volatility = mkt_env.get_const('volatility')
        self.final_date = mkt_env.get_const('final_date')
        self.currency = mkt_env.get_const('currency')
        self.frequency = mkt_env.get_const('frequency')
        self.paths = mkt_env.get_const('paths')
        self.discount_curve = mkt_env.get_curve('discount_curve') # 贴现因子
        self.time_grid = mkt_env.get_list('time_grid')
        self.special_dates = mkt_env.get_list('special_dates')
        self.instrument_values = None
        self.correlated = corr # 是否具有相关性，如果有相关性，必须进行处理
        if corr is True:
            self.cholesky_matrix = mkt_env.get_list('cholesky_matrix')
            self.rn_set = mkt_env.get_list('rn_set')[self.name]
            self.random_numbers = mkt_env.get_list('random_numbers')

    def generate_time_grid(self):
        start = self.pricing_date
        end = self.final_date
        time_grid = pd.date_rand(start=start, end=end, freq=self.frequency).to_pydatetime()
        time_grid = list(time_grid)
        if start not in time_grid:
            time_grid.insert(0, start)
        if end not in time_grid:
            time_grid.append(end)
        if len(self.special_dates)>0:
            time_grid.extend(self.special_dates)
            time_grid = list(set(time_grid))
            time_grid.sort()
        self.time_grid = np.array(time_grid)

    def get_instrument_values(self, fixed_seed=True):
        if self.instrument_values is None:
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.0)
        elif fixed_seed is False:
            self.generate_paths(fixed_seed=fixed_seed, day_count=365.0)
        return self.instrument_values

    def generate_paths(self, fixed_seed=True, day_count=365.0):
        pass