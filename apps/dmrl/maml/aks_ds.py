#
import pandas as pd
import akshare as ak

class AksDs(object):
    def __init__(self):
        self.name = 'apps.dmrl.maml.aks_ds.AksDs'

    def get_stock_dk(self, stock_symbol):
        '''
        获取日K线历史数据
        '''
        hfq_factor_df = ak.stock_zh_a_daily(symbol=stock_symbol, adjust="hfq", start_date='2020-06-25', end_date='2021-06-21')
        #print('df: {0}; {1};'.format(type(hfq_factor_df), hfq_factor_df))
        hfq_factor_df.to_csv('./data/{0}.csv'.format(stock_symbol))
        print('^_^ The End ^_^')

    def get_stocks(self):
        ''' 获取A股市场股票列表
        '''
        stocks_df = ak.stock_zh_a_spot()
        stocks_df.to_csv('./data/aks_stocks.csv')