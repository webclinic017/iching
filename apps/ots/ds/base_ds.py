#
from __future__ import print_function
import os, os.path
import datetime as dt
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from apps.ots.event.market_event import MarketEvent

class BaseDs(object):
    ''' 抽象基类，所有数据源类的共同基类，输出：Open、High、Low、Close、Volume、I持仓量，历史数据和实盘数据统一处理 '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bar(self, symbol):
        ''' 获取最新的信息条： Open、High、Low、Close、Volume、I持仓量'''
        raise NotImplementedError('必须实现此方法：get_latest_bar')

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        ''' 获取最新的N条信息条 '''
        raise NotImplementedError('必须实现此方法：get_latest_bars')

    @abstractmethod
    def get_latest_bar_dt(self, symbol):
        ''' 返回最新数据条对应的datetime对象 '''
        raise NotImplementedError('必须实现此方法：get_latest_bar_dt')

    @abstractmethod
    def get_latest_bar_value(self, symbol, val_type):
        ''' 返回最新数据条中的以val_type指定的值，如：Open、High、Low、Close、Volume、I持仓量 '''
        raise NotImplementedError('必须实现此方法：get_latest_bar_value')

    @abstractmethod
    def get_latest_bar_values(self, symbol, val_type, N=1):
        ''' 返回最新N条指定类型数据值，不足N返回实际数量 '''
        raise NotImplementedError('必须实现此方法：get_latest_bar_values')

    @abstractmethod
    def update_bars(self):
        ''' 
        将最新的数据条放入到数据序列中，采用元组形式：
        (datetiem, open, high, low, close, volume, interest)
        '''
        raise NotImplementedError('必须实现此方法：update_bars')
