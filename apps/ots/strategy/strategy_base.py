#
from __future__ import print_function
import datetime as dt
try:
    import Queue as Queue
except ImportError:
    import queue
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd

class StrategyBase(object):
    '''
    对于给定的标的代码，基于市场行情数据，生成SignalEvent。既可以
    用来历史数据也可以用来处理实际交易数据。
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def calculate_signals(self, event):
        ''' 计算信号的机制 '''
        raise NotImplementedError('必须实现calculate_signals方法')