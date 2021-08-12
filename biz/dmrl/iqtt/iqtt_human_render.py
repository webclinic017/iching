#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import time
from datetime import datetime
import pandas as pd

class IqttHumanRender(object):
    def __init__(self, title='易经量化交易系统'):
        self.name = 'biz.dmrl.iqtt.iqtt_human_render.IqttHumanRender'
        # 绘制交易历史动态图
        fig = plt.figure()
        fig.suptitle(title)
        # Create top subplot for net worth axis
        self.net_value_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)
        # Create bottom subplot for shared price/volume axis
        self.price_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.net_value_ax)
        # Create a new axis for volume which shares its x-axis with price
        self.volume_ax = self.price_ax.twinx()
        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)
        # Show the graph without blocking the rest of the program
        plt.show(block=False)

    def render(self, trades={}):
        print('### ^_^ ### {0}：bar({1}, {2}, {3}, {4}), state=(余额：{5}, 仓位：{6}, 净值：{7})'.format(
                trades['info']['current_step'], trades['obs'][0][50], trades['obs'][0][51], trades['obs'][0][52], trades['obs'][0][53],
                trades['info']['balance'], trades['info']['position'], trades['info']['net_value']
            ))
        plt.pause(0.001)
        

    def date2num(self, date):
        return date
        '''
        org_dtf = '%Y-%m-%d'
        dst_dtf = '%Y-%m-%d'
        return datetime.strptime(date, org_dtf).strftime(dst_dtf)
        '''

    def _render_net_value(self):
        pass

    








    def exp(self):
        print('测试程序 v0.0.1')
        step_range = range(0, 40)
        df = pd.read_csv('d:/zjkj/temp/Stock-Trading-Visualization/data/MSFT.csv')
        df = df.sort_values('Date')
        dates = np.array([self.date2num(x)
                          for x in df['Date'].values[step_range]])
        print(dates)