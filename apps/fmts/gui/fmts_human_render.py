#
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.pyplot import MultipleLocator
# 
from biz.dmrl.app_config import AppConfig

class FmtsHumanRender(object):
    def __init__(self, title='易经量化交易系统'):
        self.name = 'apps.fmts.gui.fmts_human_render.FmtsHumanRender'
        # 绘制交易历史动态图
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
        plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
        fig = plt.figure()
        #fig.canvas.manager.window.showMaximized()
        fig.suptitle(title)
        # Create top subplot for net worth axis
        self.net_value_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)
        # Create bottom subplot for shared price/volume axis
        self.price_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.net_value_ax)
        # Create a new axis for volume which shares its x-axis with price
        self.volume_ax = self.price_ax.twinx()
        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)
        self.first_run = True

    def render(self, trades={}):
        print('{0}：bar({1}, {2}, {3}, {4}), state=(余额：{5}, 仓位：{6}, 净值：{7})'.format(
                trades['current_step'], trades['obs'][0][50], trades['obs'][0][51], trades['obs'][0][52], trades['obs'][0][53],
                trades['balance'], trades['position'], trades['net_value']
            ))
        if trades['current_step'] > trades['window_size'] + 1:
            if self.first_run:
                figmanager = plt.get_current_fig_manager()
                figmanager.window.state('zoomed')    #最大化
                plt.show(block=False)
                self.first_run = False
            self._render_price(trades)
            self._render_volume(trades)
            self._render_trades(trades)
            self._render_net_value(trades)
            plt.pause(0.001)

    BUY_TEXT_COLOR = '#73D3CC'
    SELL_TEXT_COLOR = '#DC2C27'
    HOLD_TEXT_COLOR = '#0000FF'
    def _render_trades(self, trades):
        for item in trades['trade_history']:
            idx = item['idx']
            if item['trade_mode'] == AppConfig.TRADE_MODE_BUY:
                color = FmtsHumanRender.BUY_TEXT_COLOR
                msg = 'BUY:{0}*{1:.2f}'.format(item['quant'], item['price'])
            else:
                color = FmtsHumanRender.SELL_TEXT_COLOR
                msg = 'Sell:{0}*{1:.2f}'.format(item['quant'], item['price'])
            '''
            self.price_ax.annotate(msg, (trades['trade_dates'][idx], trades['bars']['Close'][idx]),
                    xytext=(trades['trade_dates'][idx], trades['bars']['High'][idx]),
                    color=color,
                    fontsize=8,
                    arrowprops=(dict(color=color, shrink=0.05, arrowstyle='->')))
            '''
            self.price_ax.text(
                trades['trade_dates'][idx], trades['bars']['Close'][idx], 
                msg, 
                color='w',
                size=8, rotation=45.0,
                ha="center", va="center",
                bbox=dict(boxstyle="round",ec=(1, 0.5, 0.5),fc=color,)
            )
            

    OHLC_UP = '#ff4500'
    OHLC_DOWN = '#800080'
    VOLUME_CHART_HEIGHT = 0.33
    def _render_price(self, trades):
        self.price_ax.clear()
        upper_limit = trades['window_size']
        if upper_limit > len(trades['bars']['Open']):
            upper_limit = len(trades['bars']['Open'])
        for idx in range(upper_limit):
            data = {
                'Open': trades['bars']['Open'][idx],
                'High': trades['bars']['High'][idx],
                'Low': trades['bars']['Low'][idx],
                'Close': trades['bars']['Close'][idx],
                'Volume': trades['bars']['Volume'][idx]
            }
            self.price_ax = self.draw_candlestick(self.price_ax, idx, data, FmtsHumanRender.OHLC_UP, FmtsHumanRender.OHLC_DOWN)  
        # Print the current price to the price axis
        self.price_ax.annotate('{0:.2f}'.format(trades['bars']['Close'][-1]),
            (trades['trade_dates'][-1], trades['bars']['Close'][-1]),
            xytext=(trades['trade_dates'][-1], trades['bars']['High'][-1]),
            bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
            color="black",
            fontsize="small"
        )
        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * FmtsHumanRender.VOLUME_CHART_HEIGHT, ylim[1])

    def _render_volume(self, trades): #current_step, net_worth, dates, 
                   #step_range):
        self.volume_ax.clear()
        up_idxs = []
        up_volumes = []
        down_idxs = []
        down_volumes = []
        for idx in range(len(trades['bars']['Volume'])):
            if trades['bars']['Open'][idx] - trades['bars']['Close'][idx] < 0:
                up_idxs.append(idx)
                up_volumes.append(trades['bars']['Volume'][idx])
            else:
                down_idxs.append(idx)
                down_volumes.append(trades['bars']['Volume'][idx])
        volume = np.array(trades['bars']['Volume']) # self.df['Volume'].values[step_range])
        # Color volume bars based on price direction on that date
        self.volume_ax.bar(np.array(up_idxs), np.array(up_volumes), color=FmtsHumanRender.OHLC_UP,
            alpha=0.4, width=1, align='center')
        self.volume_ax.bar(np.array(down_idxs), np.array(down_volumes), color=FmtsHumanRender.OHLC_DOWN,
            alpha=0.4, width=1, align='center')
        # Cap volume axis height below price chart and hide ticks
        self.volume_ax.set_ylim(0, max(volume) / FmtsHumanRender.VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])
        

    def draw_candlestick(self, axis, current_step, data, color_up, color_down):
        # Check if stock closed higher or not
        if data['Close'] > data['Open']:
            color = color_up
        else:
            color = color_down
        # Plot the candle wick
        axis.plot([current_step, current_step], [data['Low'], data['High']], linewidth=1.5, color='black', solid_capstyle='round', zorder=2)
        lower_y = data['Open']
        if lower_y > data['Close']:
            lower_y = data['Close']
        # Draw the candle body
        rect = mpl.patches.Rectangle((current_step - 0.25, lower_y), 0.5, abs(data['Close'] - data['Open']), facecolor=color, edgecolor='black', linewidth=1.5, zorder=3)
        # Add candle body to the axis
        axis.add_patch(rect)
        return axis

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
        

    def date2num(self, date):
        '''
        org_dtf = '%Y-%m-%d'
        dst_dtf = '%Y-%m-%d'
        return datetime.strptime(date, org_dtf).strftime(dst_dtf)
        '''
        return date
    








    def exp(self):
        print('测试程序 v0.0.1')
        step_range = range(0, 40)
        df = pd.read_csv('d:/zjkj/temp/Stock-Trading-Visualization/data/MSFT.csv')
        df = df.sort_values('Date')
        dates = np.array([self.date2num(x)
                          for x in df['Date'].values[step_range]])
        print(dates)