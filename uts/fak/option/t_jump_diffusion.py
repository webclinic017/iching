#
import datetime as dt
import unittest
from pylab import mpl, plt
import unittest
from fak.option.geometric_brownian_motion import GeometricBrownianMotion
from fak.option.jump_diffusion import JumpDiffusion
from fak.option.market_environment import MarketEnvironment
from fak.option.constant_short_rate import ConstantShortRate

class TJumpDiffusion(unittest.TestCase):
    def test_run(self):
        # 生成几何布朗运动市场环境
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
        # 生成跳跃扩散市场环境
        me_jd = MarketEnvironment('me_jd', dt.datetime(2020, 1, 1))
        me_jd.add_constant('lambda', 0.3)
        me_jd.add_constant('mu', -0.75)
        me_jd.add_constant('delta', 0.1)
        me_jd.add_environment(me_gbm)
        # 生成跳跃扩散模拟类
        jd = JumpDiffusion('jd', me_jd)
        paths_3 = jd.get_instrument_values()
        jd.update(lamb=0.9)
        paths_4 = jd.get_instrument_values()
        # 绘制图形
        plt.figure(figsize=(10, 6))
        p1 = plt.plot(gbm.time_grid, paths_3[:, :10], 'b')
        p2 = plt.plot(gbm.time_grid, paths_4[:, :10], 'r-')
        lengend1 = plt.legend([p1[0], p2[0]], ['low intensity', 'high intensity'], loc=3)
        plt.gca().add_artist(lengend1)
        plt.xticks(rotation=30)
        plt.show()