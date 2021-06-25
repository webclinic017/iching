#
import pandas as pd
import akshare as ak

class AksDs(object):
    def __init__(self):
        self.name = 'apps.dmrl.maml.aks_ds.AksDs'

    def get_stocks_dk(self, start_date, end_date):
        stock_symbols = self.get_stocks()
        total = len(stock_symbols)
        idx = 1
        for stock_symbol in stock_symbols:
            print('获取{0}日K线数据（{1}~{2}）：{3}...%'.format(stock_symbol, start_date, end_date, idx / total * 100))
            self.get_stock_dk(stock_symbol=stock_symbol, start_date=start_date, end_date=end_date)
            idx += 1

    def get_stock_dk(self, stock_symbol, start_date, end_date):
        '''
        获取日K线历史数据
        '''
        hfq_factor_df = ak.stock_zh_a_daily(symbol=stock_symbol, adjust="hfq", start_date=start_date, end_date=end_date)
        #print('df: {0}; {1};'.format(type(hfq_factor_df), hfq_factor_df))
        hfq_factor_df.to_csv('./data/aks_dks/{0}.csv'.format(stock_symbol))

    def fetch_stocks(self):
        ''' 获取A股市场股票列表
        '''
        stocks_df = ak.stock_zh_a_spot()
        stocks_df.to_csv('./data/aks_stocks.csv')

    def get_stocks(self):
        stocks_df = pd.read_csv('./data/aks_stocks.csv')
        return stocks_df.iloc[1:, 1]