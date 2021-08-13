#
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.pyplot import MultipleLocator

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
                trades['current_step'], trades['obs'][0][50], trades['obs'][0][51], trades['obs'][0][52], trades['obs'][0][53],
                trades['balance'], trades['position'], trades['net_value']
            ))
        self._render_net_value(trades)
        plt.pause(0.001)
        

    def date2num(self, date):
        '''
        org_dtf = '%Y-%m-%d'
        dst_dtf = '%Y-%m-%d'
        return datetime.strptime(date, org_dtf).strftime(dst_dtf)
        '''
        return date

    def _render_net_value(self, trades):
        # Clear the frame rendered last step
        self.net_value_ax.clear()
        # Plot net worths
        self.net_value_ax.plot_date(trades['trade_dates'], np.array(trades['net_values']), '-', label='Net Worth')
        plt.gcf().autofmt_xdate()
        ax=plt.gca()
        #ax为两条坐标轴的实例
        ax.xaxis.set_major_locator(MultipleLocator(5))
        #date_format = mpl_dates.DateFormatter('%b, %d %Y')
        #plt.gca().xaxis.set_major_formatter(date_format)
        # Show legend, which uses the label we defined for the plot above
        self.net_value_ax.legend()
        legend = self.net_value_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)
        last_date = trades['trade_dates'][-1]
        last_net_value = trades['net_values'][-1]
        # Annotate the current net worth on the net worth graph
        self.net_value_ax.annotate(
            '{0:.2f}'.format(trades['net_value']),     
            (last_date, last_net_value),
            xytext=(last_date, last_net_value),
            bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
            color="black",
            fontsize="small"
        )
        # Add space above and below min/max net worth
        self.net_value_ax.set_ylim(
            min(trades['net_values']) / 1.25,    
            max(trades['net_values']) * 1.25)
    








    def exp(self):
        print('测试程序 v0.0.1')
        step_range = range(0, 40)
        df = pd.read_csv('d:/zjkj/temp/Stock-Trading-Visualization/data/MSFT.csv')
        df = df.sort_values('Date')
        dates = np.array([self.date2num(x)
                          for x in df['Date'].values[step_range]])
        print(dates)