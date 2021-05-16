#
import datetime as dt
import unittest
from pylab import mpl, plt
from fak.option.geometric_brownian_motion import GeometricBrownianMotion
from fak.option.market_environment import MarketEnvironment
from fak.option.constant_short_rate import ConstantShortRate

class TGeometricBrownianMotion(unittest.TestCase):
    def test_run(self):
        plt.style.use('seaborn')
        mpl.rcParams['font.family'] = 'serif'
        # 生成市场环境
        me_gbm = MarketEnvironment('me_gbm', dt.datetime(2020, 1, 1))
        me_gbm.add_constant('initial_value', 36.0)
        me_gbm.add_constant('volatility', 0.2)
        me_gbm.add_constant('final_date', dt.datetime(2020, 12, 31))
        me_gbm.add_constant('currency', 'EUR')
        me_gbm.add_constant('frequency', 'M')
        me_gbm.add_constant('paths', 1000)
        csr = ConstantShortRate('csr', 0.06)
        me_gbm.add_curve('discount_curve', csr)
        # 生成几何布朗运动模拟类
        gbm = GeometricBrownianMotion('gbm', me_gbm)
        gbm.generate_time_grid()
        print('时间节点：{0};'.format(gbm.time_grid))
        paths_1 = gbm.get_instrument_values()
        print('paths_1: {0};'.format(paths_1.round(3)))
        gbm.update(volatility=0.5)
        paths_2 = gbm.get_instrument_values()
        # 可视化结果
        plt.figure(figsize=(10, 6))
        p1 = plt.plot(gbm.time_grid, paths_1[:, :10], 'b')
        p2 = plt.plot(gbm.time_grid, paths_2[:, :10], 'r-')
        legend1 = plt.legend([p1[0], p2[0]], ['low volatility', 'high volatility'], loc=2)
        plt.gca().add_artist(legend1)
        plt.xticks(rotation=30)
        plt.show()