#
from __future__ import print_function
import numpy as np
import pandas as pd

class Performance(object):
    ''' 投资绩效评价类 '''
    @staticmethod
    def calculate_sharpe_ratio(returns, periods=252):
        ''' 计算策略的Sharpe比率，基于基准利率为0（无风险利率） '''
        print('############# returns: {0};'.format(returns))
        return np.sqrt(periods)*(np.mean(returns) / np.std(returns))

    @staticmethod
    def calculate_drawdowns(pnl):
        '''
        计算pnl（profit and loss）曲线的最大回撤（从最大收益到最小收益之间的距离），以及回撤时间
        pnl为pandas的series
        '''
        hwm = [0]
        idx = pnl.index
        drawdown = pd.Series(index=idx)
        duration = pd.Series(index=idx)
        for t in range(1, len(idx)):
            hwm.append(max(hwm[t-1], pnl[t]))
            drawdown[t] = (hwm[t]-pnl[t])
            duration[t] = (0 if drawdown[t]==0 else duration[t-1] + 1)
        return drawdown, drawdown.max(), duration.max()
