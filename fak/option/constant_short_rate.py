#
import numpy as np
from fak.option.time_util import TimeUtil

class ConstantShortRate(object):
    def __init__(self, name, short_rate):
        ''' Class for constant short rate discounting.
        Attributes
        ==========
        name: string
            name of the object
        short_rate: float (positive)
            constant rate for discounting
        Methods
        =======
        get_discount_factors:
            get discount factors given a list/array of datetime objects
            or year fractions
        '''
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0:
            raise ValueError('Short rate negative.')
            # this is debatable given recent market realities

    def get_discount_factors(self, date_list, dtobjects=True):
        if dtobjects is True:
            dlist = TimeUtil.get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)
        dflist = np.exp(self.short_rate * -dlist)
        return np.array((date_list, dflist)).T