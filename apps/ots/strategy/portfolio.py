#
from __future__ import print_function
import datetime as dt
import math
from apps.ots.strategy.performance import Performance
try:
    import Queue as queue
except ImportError:
    import queue
import numpy as np
import pandas as pd
#
from apps.ots.event.ots_event import OtsEvent
from apps.ots.event.signal_event import SignalEvent
from apps.ots.event.order_event import OrderEvent
from apps.ots.event.fill_event import FillEvent
from apps.ots.strategy.naive_risk_manager import NaiveRiskManager
from apps.ots.order.ots_order import OtsOrder
from apps.ots.ots_util import OtsUtil

class Portfolio(object):
    ''' 
    处理所有持仓和市值，其接收Stategy产生的SignalEvent，根据现有持仓情况和规则，
    以及风控模型，还有就是算法交易情况（如交易量巨大时分步来进行），生成OrderEvent，
    同时接收Exchange系统产生的FillEvent，最终更新持仓情况。
    position DataFrame 存放用时间为索引的持仓数量
    holdings DataFrame 存放特定时间索引对应的每个代码的现金和总的市场持仓价值，
    以及资产组合总量的百分比变化
    '''
    def __init__(self, bars, events, start_date, initial_capital=100000):
        self.risk_manager = NaiveRiskManager()
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital
        self.all_positions = self.construct_all_positions()
        self.current_positions = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()

    def construct_all_positions(self):
        '''
        使用start_date确定开始日期，构造持仓列表
        '''
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['datetime'] = self.start_date
        return [d]

    def construct_all_holdings(self):
        '''
        构造字典保存所有资产组合的开始日期的价值
        '''
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]
    
    def construct_current_holdings(self):
        ''' 构造这典保存所有资产组合的当前价值 '''
        d = dict((k, v) for k, v in [(s, 0.0) for s in self.symbol_list])
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_timeindex(self, event):
        '''
        根据当前市场数据，增加一条新数据，反映当前所有持仓的市场价值
        event为市场事件
        '''
        latest_datetime = self.bars.get_latest_bar_dt(self.symbol_list[0])
        dp = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        dp['datetime'] = latest_datetime
        for s in self.symbol_list:
            dp[s] = self.current_positions[s]
        self.all_positions.append(dp)
        dh = dict((k, v) for k, v in [(s, 0) for s in self.symbol_list])
        dh['datetime'] = latest_datetime
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        dh['total'] = self.current_holdings['total']
        for s in self.symbol_list:
            market_value = self.current_positions[s] * self.bars.get_latest_bar_value(s, 'adj_close')
            dh[s] = market_value
            dh['total'] += market_value
        self.all_holdings.append(dh)

    def update_positions_from_fill(self, fillEvent):
        '''
        根据FillEvent更新持仓矩阵来反映最新持仓
        '''
        fill_direction = 0
        if fillEvent.direction == 'BUY':
            fill_direction = 1
        elif fillEvent.direction == 'SELL':
            fill_direction = -1
        self.current_positions[fillEvent.symbol] += fill_direction * fillEvent.quantity

    def update_holdings_from_fill(self, fillEvent):
        '''
        根据FillEvent更新持仓价值矩阵
        '''
        fill_direction = 0
        if fillEvent.direction == 'BUY':
            fill_direction = 1
        elif fillEvent.direction == 'SELL':
            fill_direction = -1
        fill_price = self.bars.get_latest_bar_value(fillEvent.symbol, 'adj_close')
        amount = fill_direction * fill_price * fillEvent.quantity
        self.current_holdings[fillEvent.symbol] += amount
        self.current_holdings['commission'] = fillEvent.commission
        self.current_holdings['cash'] -= (amount + fillEvent.commission)
        self.current_holdings['total'] -= (amount + fillEvent.commission)

    def update_fill(self, event):
        '''
        接收到FillEvent后更新持仓和市值
        '''
        self.update_positions_from_fill(event)
        self.update_holdings_from_fill(event)

    def generate_order(self, signalEvent):
        '''
        根据信号事件生成订单对象
        '''
        order_event = None
        symbol = signalEvent.symbol
        direction = signalEvent.direction
        strength = signalEvent.strength
        mkt_quantity = self.risk_manager.get_mkt_quantity(signalEvent=signalEvent)
        curr_quantity = self.current_holdings[symbol]
        order_type = OtsOrder.OT_MKT
        if direction == 'LONG' and curr_quantity == 0:
            order_event = OrderEvent(symbol, order_type, mkt_quantity, 'BUY')
        elif direction == 'SHORT' and curr_quantity >= mkt_quantity:
            order_event = OrderEvent(symbol, order_type, mkt_quantity, 'SELL')
        elif direction == 'EXIT' and curr_quantity>0:
            order_event = OrderEvent(symbol, order_type, abs(mkt_quantity), 'SELL')
        elif direction == 'EXIT' and curr_quantity<0:
            order_event = OrderEvent(symbol, order_type, abs(mkt_quantity), 'BUY')
        return order_event

    def update_signal(self, event):
        order_event = self.generate_order(event)
        self.events.put(order_event)

    def update_from_event(self, event):
        ''' update_signal
        '''
        if event.type == OtsEvent.ET_SIGNAL:
            self.update_signal(event)
        elif event.type == OtsEvent.ET_FILL:
            self.update_fill(event)

    def create_equity_curve_dataframe(self):
        '''
        基于all_holdings的DataFrame对象
        '''
        curve = pd.DataFrame(self.all_holdings)
        print('curve: {0}; {1};'.format(type(curve), curve))
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        self.equity_curve = curve

    def output_summary_stats(self):
        '''
        所有资产组合的合计信息
        '''
        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']
        sharpe_ratio = Performance.calculate_sharpe_ratio(returns)
        drawdown, max_dd, dd_duration = Performance.calculate_drawdowns(pnl)
        self.equity_curve['drawdown'] = drawdown
        stats = [('Total Return', '{0:0.2f}%'.format((total_return - 1.0)*100)),
            ('Sharpe Ratio', '{0:0.2f}'.format(sharpe_ratio)),
            ('Max Drawdown', '{0:0.2f}'.format(max_dd*100)),
            ('Drawdown Duration', '{0}'.format(dd_duration))
        ]
        self.equity_curve.to_csv('equity.csv')
        return stats
