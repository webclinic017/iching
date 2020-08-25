# 上证50ETF股指行情数据源类
import akshare as ak

class Sh50etfIndexDataSource(object):
    def __init__(self):
        self.refl = 'apps.sop.ds.Sh50etfIndexDataSource'
        self.symbol = 'sh510050' # 50ETF指数代码

    def get_daily_data(self, start_date, end_date):
        df = ak.stock_zh_index_daily(symbol="sh510050")
        return df.loc[start_date: end_date]