#
import datetime as dt
import numpy as np
import pandas as pd
from pandas.io.data import DataReader

class DsUtil(object):
    @staticmethod
    def create_lagged_series(symbol, start_date, end_date, lags=5):
        ts = DataReader(
            symbol, 'yahoo',
            start_date - dt.timedelta(days=365),
            end_date
        )
        tslag = pd.DataFrame(index = ts.index)
        tslag['Today'] = ts['Adj Close']
        tslag['Volume'] = ts['Volume']
        for i in range(0, lags):
            tslag['Lag{0}'.format(str(i+1))] = ts['Adj Close'].shift(i+1)
        tsret = pd.DataFrame(index = ts.index)
        tsret['Volume'] = tslag['Volume']
        tsret['Today'] = tslag['Today'].pct_change()*100.0
        # 避免机器学习中为0的情况
        for i, x in enumerate(tsret['Today']):
            if (abs(x)<0.0001):
                tsret['Today'][i] = 0.0001
        for i in range(0, lags):
            tsret['lag{0}'.format(str(i+1))] = tslag['lag{0}'.format(str(i+1))].pct_change() * 100.0
        tsret['Direction'] = np.sign(tsret['Today'])
        tsret = tsret[tsret.index>=start_date]
        return tsret