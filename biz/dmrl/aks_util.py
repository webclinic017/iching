# AKS工具类，用于生成和整理数据集
import numpy as np
import matplotlib.pyplot as plt
#
from biz.dmrl.app_config import AppConfig

class AksUtil(object):
    @staticmethod
    def load_minute_bar_ds(stock_symbol):
        '''
        读入分钟级行情，格式为：day,open,high,low,close,volume， 对所有数据进行对数收益率处理，并
        进行归一化处理（减均值除标准差），并且同时取出开盘、最高、最低、收盘价的绝对数值
        '''
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
        return (ds-ds_mu)/ds_std, raw_ds[1:, 0:4]

    @staticmethod
    def generate_stock_ds(stock_symbol, ds_mode=0, draw_line=False):
        '''
        生成指定股票的数据集，强化学习环境中的数据集最后4列为价格的绝对数值
            ds_mode 模式：0-用于MAML训练；1-用于强化学习环境执行；
            draw_line false-不绘制；true-绘制；
        返回值：
            X：样本集，[-1, 50]，前10天的数据；若ds_mode=1：[-1, 90]，前10天的数据；
        '''
        print('生成训练数据集')
        if draw_line:
            plt.ion()
            fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        s1_ds, prices = AksUtil.load_minute_bar_ds(stock_symbol)
        total_samples = len(s1_ds) # total_samples = idx + forward_step
        # 生成第一个样本
        idx = AppConfig.mdp_params['back_window']
        X1_raw = []
        y1_raw = []
        for idx in range(AppConfig.mdp_params['back_window'], total_samples - AppConfig.mdp_params['forward_step']):
            print('第{0}步：...'.format(idx-AppConfig.mdp_params['back_window']+1))
            raw_data = s1_ds[idx - AppConfig.mdp_params['back_window'] : idx]
            prices_data = prices[idx]
            sample = raw_data.reshape((raw_data.shape[0]*raw_data.shape[1], ))
            if 1 == ds_mode:
                sample = np.append(sample, prices_data) # 用于强化学习环境
            X1_raw.append(sample)
            y1_raw.append(AksUtil.get_market_regime(prices[idx : idx + AppConfig.mdp_params['forward_step']][3]))
            if draw_line:
                AksUtil.draw_line_chart(prices[idx : idx + AppConfig.mdp_params['forward_step']][3])
        X1 = np.array(X1_raw)
        y1 = np.array(y1_raw)
        if draw_line:
            plt.show(block=True)
        return X1, y1

        
    @staticmethod
    def get_market_regime(data):
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

    @staticmethod
    def draw_line_chart(data):
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