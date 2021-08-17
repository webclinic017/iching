# Akshare数据下载和处理程序
import akshare as ak

class AkshareDataSource(object):
    def __init__(self):
        self.name = 'apps.fmts.ds.akshare_data_source.AkshareDataSource'

    @staticmethod
    def get_minute_bars(stock_symbol, period='1', adjust='hfq'):
        '''
        调用AkshareDataSource接口，默认为1分钟，复权后数据，保持历史价格不变，当配股、分拆、合并、派发股息后价格会变化，
        这种方式虽然不利于看盘，但是能反映真实收益率，量化交易研究中通常采用。
        '''
        return ak.stock_zh_a_minute(symbol=stock_symbol, period=period, adjust=adjust)