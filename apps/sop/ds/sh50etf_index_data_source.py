# 上证50ETF股指行情数据源类
import akshare as ak

class Sh50etfIndexDataSource(object):
    def __init__(self):
        self.refl = 'apps.sop.ds.Sh50etfIndexDataSource'
        self.symbol = 'sh510050' # 50ETF指数代码

    def get_daily_data(self, start_date, end_date):
        df = ak.stock_zh_index_daily(symbol="sh510050")
        df1 = df.loc[start_date: end_date]
        print('')
        print(df1)
        open1 = df1['open'][start_date]
        print('open1: {0};'.format(type(open1), open1))
        val1 = df1.loc[start_date]
        print('df1[2020-06-01]: {0}; {1};'.format(type(val1), val1))
        print('open: {0}; high: {1};  type:{2}'.format(val1['open'], val1['high'], type(val1['open'])))