#
import pandas as pd
import datetime
import tushare as ts
import numpy as np
from math import log,sqrt,exp
from scipy import stats

class OtsQuotation(object):
    def __init__(self):
        self.name = ''
        self.pro = ts.pro_api()

    def startup(self):
        df = self.extract_data()
        print(df)

    def extract_data(self, date): # 提取数据
        # 提取50ETF合约基础信息
        df_basic = self.pro.opt_basic(exchange='SSE', fields='ts_code,name,call_put,exercise_price,list_date,delist_date')
        df_basic = df_basic.loc[df_basic['name'].str.contains('50ETF')]
        df_basic = df_basic[(df_basic.list_date<=date)&(df_basic.delist_date>date)] # 提取当天市场上交易的期权合约
        df_basic = df_basic.drop(['name','list_date'],axis=1)
        df_basic['date'] = date

        # 提取日线行情数据
        df_cal = self.pro.trade_cal(exchange='SSE', cal_date=date, fields = 'cal_date,is_open,pretrade_date')
        if df_cal.iloc[0, 1] == 0:
            date = df_cal.iloc[0, 2] # 判断当天是否为交易日，若否则选择前一个交易日

        opt_list = df_basic['ts_code'].tolist() # 获取50ETF期权合约列表
        df_daily = self.pro.opt_daily(trade_date=date,exchange = 'SSE',fields='ts_code,trade_date,settle')
        df_daily = df_daily[df_daily['ts_code'].isin(opt_list)]

        # 提取50etf指数数据
        df_50etf = self.pro.fund_daily(ts_code='510050.SH', trade_date = date,fields = 'close')
        s = df_50etf.iloc[0, 0] 

        # 提取无风险利率数据（用一周shibor利率表示）
        df_shibor = self.pro.shibor(date = date,fields = '1w')
        rf = df_shibor.iloc[0,0]/100

        # 数据合并
        df = pd.merge(df_basic,df_daily,how='left',on=['ts_code'])
        df['s'] = s
        df['r'] = rf
        df = df.rename(columns={'exercise_price':'k', 'settle':'c'})
        #print(df)
        return df