#
import pandas as pd
import akshare as ak

class AksDs(object):
    def __init__(self):
        self.name = 'apps.dmrl.maml.aks_ds.AksDs'

    def get_minute_bar(self, stock_symbol, period = '1', adjust='hfq'):
        '''
        获取指定股票分钟级1分钟级别复权后行情数据
        stock_symbol 股票代码
        period 默认为1分钟级别，可以取1、5、15分钟级别
        adjust 可选qfq复权前，hfq复权后
        '''
        return ak.stock_zh_a_minute(symbol=stock_symbol, period=period, adjust=adjust)

    def get_high_correlate_stocks(self):
        '''
        '''
        idx = 1
        stocks = {}
        with open('./data/aks_corrs.txt', 'r', encoding='utf-8') as fd:
            for row in fd:
                row = row.strip()
                arrs = row.split(':')
                key = arrs[0]
                if 'sh688690' in key or 'sz001207' in key or 'sh688216' in key:
                    continue
                val = float(arrs[2])
                arrs2 = arrs[0].split('-')
                s1 = arrs2[0]
                s2 = arrs2[1]
                if val > 0.97 and '{0}-{1}'.format(s1, s2) not in stocks and '{0}-{1}'.format(s2, s1) not in stocks:
                    stocks[key] = val
                idx += 1
                if idx % 1000 == 0:
                    print('已经处理{0}条记录，获取{1}个股票对...'.format(idx, len(stocks)))
        stock_nums = {}
        for k, v in stocks.items():
            arrs = k.split('-')
            s1 = arrs[0]
            s2 = arrs[1]
            if s1 in stock_nums:
                stock_nums[s1] += 1
            else:
                stock_nums[s1] = 1
            if s2 in stock_nums:
                stock_nums[s2] += 1
            else:
                stock_nums[s2] = 1
        sel_stocks = set()
        for k, v in stock_nums.items():
            print('{0}={1};'.format(k, v))
            if v >= 5:
                sel_stocks.add(k)
        with open('./data/aks_pairs.txt', 'w', encoding='utf-8') as fd:
            for k in sorted(stocks):
                v = stocks[k]
                arrs = k.split('-')
                s1 = arrs[0]
                s2 = arrs[1]
                if s1 in sel_stocks and s2 in sel_stocks:
                    fd.write('{0}:{1}\n'.format(k, v))

    def calculate_corrs(self):
        stock1s = self.get_stocks()
        stock2s = self.get_stocks()
        len1 = len(stock1s)
        total = len1 * len1
        corr_dict = {}
        idx = 1
        for stock1 in stock1s:
            for stock2 in stock2s:
                if stock1 != stock2:
                    dk1 = pd.read_csv('./data/aks_dks/{0}.csv'.format(stock1))
                    x = dk1.iloc[0:, 4]
                    dk2 = pd.read_csv('./data/aks_dks/{0}.csv'.format(stock2))
                    y = dk2.iloc[0:, 4]
                    corr_dict['{0}-{1}: '.format(stock1, stock2)] = x.corr(y)
                    print('progress {0}: {1}/{2};'.format(idx/total*100, idx, total))
                    idx += 1
        with open('./data/aks_corrs.txt', 'w', encoding='utf-8') as fd:
            for k, v in corr_dict.items():
                print('### {0}: {1};'.format(k, v))
                fd.write('{0}:{1}\n'.format(k, v))

    def calculate_corr(self, stock1, stock2):
        '''
        计算stock1和stock2两支股票的相关性correlation，为1时代表正相关，为0时代表负相关
        '''
        stock1_df = pd.read_csv('./data/aks_dks/{0}.csv'.format(stock1))
        x = stock1_df.iloc[0:, 4]
        stock2_df = pd.read_csv('./data/aks_dks/{0}.csv'.format(stock2))
        y = stock2_df.iloc[0:, 4]
        x_y_corr = x.corr(y)
        print('corr_{0}_{1}: {2};'.format(stock1, stock2, x_y_corr))

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
        return stocks_df.iloc[0:, 1]