#
import datetime as dt
import unittest
from pylab import mpl, plt
import unittest
from fak.option.geometric_brownian_motion import GeometricBrownianMotion
from fak.option.market_environment import MarketEnvironment
from fak.option.constant_short_rate import ConstantShortRate
from fak.option.squre_root_diffusion import SquareRootDiffusion

class TSquareRootDiffusion(unittest.TestCase):
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
        me_srd = MarketEnvironment('me_srd', dt.datetime(2020, 1, 1))
        me_srd.add_constant('initial_value', .25)
        me_srd.add_constant('volatility', 0.05)
        me_srd.add_constant('final_date', dt.datetime(2020, 12, 31))
        me_srd.add_constant('currency', 'EUR')
        me_srd.add_constant('frequency', 'W')
        me_srd.add_constant('paths', 10000)
        # specific to simualation class
        me_srd.add_constant('kappa', 4.0)
        me_srd.add_constant('theta', 0.2)
        me_srd.add_curve('discount_curve', ConstantShortRate('r', 0.0))
        srd = SquareRootDiffusion('srd', me_srd)
        srd_paths = srd.get_instrument_values()[:, :10]
        plt.figure(figsize=(10, 6))
        plt.plot(srd.time_grid, srd.get_instrument_values()[:, :10])
        plt.axhline(me_srd.get_constant('theta'), color='r',
                    ls='--', lw=2.0)
        plt.xticks(rotation=30)
        plt.show()