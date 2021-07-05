#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import akshare as ak
#
from apps.dmrl.maml.app_config import AppConfig

class AksDs(object):

    def __init__(self):
        self.name = 'apps.dmrl.maml.aks_ds.AksDs'

    def generate_stock_ds(self, stock_symbol, draw_line=False):
        print('生成训练数据集')
        if draw_line:
            plt.ion()
            fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        s1_ds, close_prices = self.load_minute_bar_ds(stock_symbol)
        total_samples = len(s1_ds) # total_samples = idx + forward_step
        # 生成第一个样本
        idx = AppConfig.mdp_params['back_window']
        X1_raw = []
        y1_raw = []
        for idx in range(AppConfig.mdp_params['back_window'], total_samples - AppConfig.mdp_params['forward_step']):
            print('第{0}步：...'.format(idx-AppConfig.mdp_params['back_window']+1))
            raw_data = s1_ds[idx - AppConfig.mdp_params['back_window'] : idx]
            sample = raw_data.reshape((raw_data.shape[0]*raw_data.shape[1], ))
            X1_raw.append(sample)
            y1_raw.append(self.get_market_regime(close_prices[idx : idx + AppConfig.mdp_params['forward_step']]))
            if draw_line:
                self.draw_line_chart(close_prices[idx : idx + AppConfig.mdp_params['forward_step']])
        X1 = np.array(X1_raw)
        y1 = np.array(y1_raw)
        if draw_line:
            plt.show(block=True)
        return X1, y1

    def get_market_regime(self, data):
        '''
        对于分钟级K线数据，取当前点之后1小时数据，如果数据首先超过上限，则识别为上升行情，
        如果数据首先超过下限，则为下跌行情，否则为震荡行情。有了市场状态之后，可以根据持
        仓情况，决定适合的操作。
        '''    
        c_mu = np.mean(data)
        c_std = np.std(data) 
        if data[0] + AppConfig.mr_params['asc_span_coff'] * c_std > (1+AppConfig.mr_params['asc_threshold'])*data[0]:
            asc_delta = AppConfig.mr_params['asc_span_coff'] * c_std
        else:
            asc_delta = AppConfig.mr_params['asc_threshold']*data[0]
        if data[0] - AppConfig.mr_params['desc_span_coff'] * c_std > (1-AppConfig.mr_params['desc_threshold']) * data[0]:
            desc_delta = AppConfig.mr_params['desc_threshold'] * data[0]
        else:
            desc_delta = AppConfig.mr_params['desc_span_coff'] * c_std
        market_regime = AppConfig.MR_VIBRATE
        cnt = len(data)
        for i in range(1, cnt, 1):
            if data[i] > data[0] + asc_delta:
                market_regime = AppConfig.MR_BULL
                break
            if data[i] < data[0] - desc_delta:
                market_regime = AppConfig.MR_BEAR
                break
        return market_regime

    def draw_line_chart(self, data):
        '''
        绘制一维折线图，并标记出上限和下限
        data 一维数据
        '''
        asc_span_coff = 0.5 # std的变化系数，越小则表明越容易出现上涨或下跌模式
        desc_span_coff = 0.3
        c_mu = np.mean(data)
        c_std = np.std(data) 
        asc_threshold = 0.008
        desc_threshold = 0.006
        #asc_delta = asc_span_coff * c_std
        #desc_delta = desc_span_coff * c_std
        if data[0] + asc_span_coff * c_std > (1+asc_threshold)*data[0]:
            asc_delta = asc_span_coff * c_std
        else:
            asc_delta = asc_threshold*data[0]
        if data[0] - desc_span_coff * c_std > (1-desc_threshold) * data[0]:
            desc_delta = desc_threshold * data[0]
        cnt = data.shape[0] # 数据总点数作为横坐标
        y0 = np.ones((cnt,), dtype=np.float32) * (data[0] - desc_delta) # 超过此限认为是下跌趋势
        y1 = np.ones((cnt,), dtype=np.float32) * data[0]
        y2 = np.ones((cnt,), dtype=np.float32) * (data[0] + asc_delta) # 超过此限认为是上涨趋势
        x = range(cnt)
        plt.plot(x, data, marker='*')
        plt.plot(x, y0)
        plt.plot(x, y1)
        plt.plot(x, y2)
        x2 = np.array([cnt-1, cnt-1])
        yr = np.array([data[0] - desc_delta, data[0] + asc_delta])
        plt.plot(x2, yr)
        x3 = np.array([0.0, 0.0])
        plt.plot(x3, yr)
        #plt.show()
        plt.draw()
        plt.pause(0.1)
        plt.cla()

    def load_minute_bar_ds(self, stock_symbol):
        csv_file = './data/aks_1ms/{0}_1m.csv'.format(stock_symbol)
        items = []
        with open(csv_file, 'r', encoding='utf-8') as fd:
            is_first_row = True
            for row in fd:
                if is_first_row:
                    is_first_row = False
                    continue
                row = row.strip()
                arrs0 = row.split(',')
                if len(row)<=0 or arrs0[1]=='' or arrs0[2]=='' or arrs0[3]=='' or arrs0[4]=='' or arrs0[5]=='':
                    break
                item = []
                item.append(float(arrs0[1]))
                item.append(float(arrs0[2]))
                item.append(float(arrs0[3]))
                item.append(float(arrs0[4]))
                item.append(float(arrs0[5]))
                items.append(item)
        raw_ds = np.array(items, dtype=np.float32)
        log_ds = np.log(raw_ds)
        ds = np.diff(log_ds, n=1, axis=0)
        ds_mu = np.mean(ds, axis=0)
        ds_std = np.std(ds, axis=0)
        return (ds-ds_mu)/ds_std, raw_ds[:, 3]

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