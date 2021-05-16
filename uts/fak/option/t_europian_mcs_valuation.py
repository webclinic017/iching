#
import datetime as dt
import unittest
import numpy as np
from fak.option.europian_mcs_valuation import EuropianMcsValuation
from fak.option.market_environment import MarketEnvironment
from fak.option.constant_short_rate import ConstantShortRate
from fak.option.geometric_brownian_motion import GeometricBrownianMotion
from fak.option.option_stats_ploter import OptionStatsPloter

class TEuropianMcsValuation(unittest.TestCase):
    def test_run(self):
        # 生成几何布朗运动模拟市场环境
        me_gbm = MarketEnvironment('me_gbm', dt.datetime(2020, 1, 1))
        me_gbm.add_constant('initial_value', 36.)
        me_gbm.add_constant('volatility', 0.2)
        me_gbm.add_constant('final_date', dt.datetime(2020, 12, 31))
        me_gbm.add_constant('currency', 'EUR')
        me_gbm.add_constant('frequency', 'M')
        me_gbm.add_constant('paths', 10000)
        csr = ConstantShortRate('csr', 0.06)
        me_gbm.add_curve('discount_curve', csr)
        # 生成几何布朗运动模拟类
        gbm = GeometricBrownianMotion('gbm', me_gbm)
        # 生成欧式看涨期权市场环境
        me_call = MarketEnvironment('me_call', me_gbm.pricing_date)
        me_call.add_constant('strike', 40.)
        me_call.add_constant('maturity', dt.datetime(2020, 12, 31))
        me_call.add_constant('currency', 'EUR')
        payoff_func = 'np.maximum(maturity_value - strike, 0)'
        # 生成欧式看涨期权估值类
        eur_call = EuropianMcsValuation('eur_call', underlying=gbm,
                        mar_env=me_call, payoff_func=payoff_func)
        # 估值结果
        val = eur_call.present_value()
        delta = eur_call.delta()
        vega = eur_call.vega()
        print('val={0}; delta={1}; vega={2};'.format(val, delta, vega))
        # 绘制图表
        s_list = np.arange(34., 46.1, 2.)
        p_list = []; d_list = []; v_list = []
        for s in s_list:
            eur_call.update(initial_value=s)
            p_list.append(eur_call.present_value(fixed_seed=True))
            d_list.append(eur_call.delta())
            v_list.append(eur_call.vega())
        OptionStatsPloter.plot_option_stats(s_list, p_list, d_list, v_list)