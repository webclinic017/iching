import datetime as dt
import numpy as np

class DtUtil(object):
    def __init__(self):
        self.refl = ''

    @staticmethod
    def get_year_deltas(date_list, day_count=365.0):
        start = date_list[0]
        delta_list = [ (curr-start).days / day_count for curr in date_list]
        return np.array(delta_list)