#
import numpy as np

class TimeUtil(object):
    def get_year_deltas(self, date_list: list, day_count: float=365.) -> 'np.ndarray':
        ''' Return vector of floats with day deltas in year fractions.
        Initial value normalized to zero.
        Parameters
        ==========
        date_list: list or array
            collection of datetime objects
        day_count: float
            number of days for a year
            (to account for different conventions)
        Results
        =======
        delta_list: array
            year fractions
        '''
        start = min(date_list)
        delta_list = [(date - start).days / day_count
                    for date in date_list]
        return np.array(delta_list)