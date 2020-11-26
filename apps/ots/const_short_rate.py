#
import numpy as np
from apps.ots.dt_util import DtUtil

class ConstShortRate(object):
    ''' 短期利率贴现类 '''
    def __init__(self, name, short_rate):
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0:
            raise ValueError('短期利率不能为负')

    def get_discount_factor(self, date_list, dtobject=True):
        if dtobject is True:
            dlist = DtUtil.get_year_deltas(date_list)
        else:
            dlist = np.array(date_list)
        dflist = np.exp(self.short_rate * np.sort(-dlist))
        return np.array((date_list, dflist)).T