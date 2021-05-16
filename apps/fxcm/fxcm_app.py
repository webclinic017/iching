#
import time
import numpy as np
import pandas as pd
import datetime as dt
from pylab import mpl, plt
import fxcmpy
from fxcmpy import fxcmpy_tick_data_reader as tdr


'''
fxcm网址：https://tradingstation.fuhuisupport.com/?utm_source=sfmc&utm_medium=email&utm_campaign=Markets_demo_confirmation_ts2_SC&utm_term=https%3a%2f%2ftradingstation.fuhuisupport.com%2f&utm_id=744746&sfmc_id=269578845
用户名：D103428644
密码：G9koa
token: 26587486829c9e848a507f4d85462c793b05c5e8
'''

class FxcmApp(object):
    def __init__(self):
        self.name = 'apps.fxcm.fxcm_app.FxcmApp'

    def startup(self, args={}):
        print('算法交易平台')
        api = fxcmpy.fxcmpy(config_file='./apps/fxcm/config/fxcm.config')
        print('api is ok')
        plt.style.use('seaborn')
        mpl.rcParams['font.family'] = 'serif'
        print(tdr.get_available_symbols())
        #
        start = dt.datetime(2018, 6, 25)  
        stop = dt.datetime(2018, 6, 30)
        td = tdr('EURUSD', start, stop)
        print(td.get_raw_data().info())
        print(td.get_data().info())
        print(td.get_data().head())
        sub = td.get_data(start='2018-06-29 12:00:00',
                  end='2018-06-29 12:15:00')
        print('sub: {0};'.format(sub))
        sub['Mid'] = sub.mean(axis=1)
        sub['SMA'] = sub['Mid'].rolling(1000).mean()
        sub[['Mid', 'SMA']].plot(figsize=(10, 6), lw=0.75)
        plt.show()