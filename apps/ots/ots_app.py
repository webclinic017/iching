#
import datetime as dt
import matplotlib.pyplot as plt
from apps.ots.dt_util import DtUtil
from apps.ots.nc_util import NcUtil
from apps.ots.const_short_rate import ConstShortRate
from apps.ots.market_environment import MarketEnvironment
from apps.ots.simulation_base import SimulationBase
from apps.ots.geometric_browning_motion import GeometricBrowningMotion
from apps.ots.jump_diffusion import JumpDiffusion
from apps.ots.square_root_diffusion import SquareRootDiffusion
#from apps.ots.valuation.evaluation_mcs_eu import EvaluationMcsEu
from apps.ots.exp.volatility_arbitrage import VolatilityArbitrage

class OtsApp(object):
    def __init__(self):
        self.refl = ''

    def exp001(self):
        ''' 短期利率贴现因子 '''
        dates = [dt.datetime(2019, 1, 1), dt.datetime(2019, 7, 1), dt.datetime(2020, 1, 1)]
        csr = ConstShortRate('n1', 0.05)
        dfs = csr.get_discount_factor(dates)
        print(dfs)

        deltas = DtUtil.get_year_deltas(dates)
        dfs1 = csr.get_discount_factor(deltas, dtobject=False)
        print(dfs1)

    def exp002(self):
        dates = [dt.datetime(2019, 1, 1), dt.datetime(2019, 7, 1), dt.datetime(2020, 1, 1)]
        csr = ConstShortRate('n1', 0.05)
        me_1 = MarketEnvironment('me_1', dt.datetime(2019, 1, 1))
        me_1.add_list('symbols', ['APPL', 'MSFT', 'FB'])
        print(me_1.get_list('symbols'))
        me_2 = MarketEnvironment('me_2', dt.datetime(2019, 1, 1))
        me_2.add_const('volatility', 0.2)
        me_2.add_curve('short_rate', csr)
        print(me_2.get_curve('short_rate'))
        me_1.add_environment(me_2)
        print(me_1.get_curve('short_rate').short_rate)

    def exp003(self):
        num1 = NcUtil.gen_random_numbers((2, 2, 2), antithetic=False, moment_matching=False)
        print(num1)
        num2 = NcUtil.gen_random_numbers((2, 3, 2), antithetic=False, moment_matching=True)
        print(num2)
        print('mean: {0}; std: {1};'.format(num2.mean(), num2.std()))

    def exp004(self):
        me_gbm = MarketEnvironment('me_gbm', dt.datetime(2019, 1, 1))
        me_gbm.add_const('initial_value', 36.0)
        me_gbm.add_const('volatility', 0.2)
        me_gbm.add_const('final_date', dt.datetime(2019, 12, 31))
        me_gbm.add_const('currency', 'RMB')
        me_gbm.add_const('frequency', 'M')
        me_gbm.add_const('paths', 10000)
        csr = ConstShortRate('csr', 0.05)
        me_gbm.add_curve('discount_curve', csr)
        gbm = GeometricBrowningMotion('gbm', me_gbm)

        gbm.generate_time_grid()
        print(gbm.time_grid)
        gbm.generate_paths()
        paths_1 = gbm.instrument_values
        print(gbm.instrument_values)
        gbm.update(volatility=0.5)
        gbm.generate_paths()
        paths_2 = gbm.instrument_values
        print(gbm.instrument_values)
        #
        plt.figure(figsize=(8, 4))
        p1 = plt.plot(gbm.time_grid, paths_1[:, :10], 'b')
        p2 = plt.plot(gbm.time_grid, paths_2[:, :10], 'r-.')
        plt.grid(True)
        l1 = plt.legend([p1[0], p2[0]], ['low_volatility', 'high_volatility'], loc=2)
        plt.gca().add_artist(l1)
        plt.xticks(rotation=30)
        plt.show()

    def exp005(self):
        me_jd = MarketEnvironment('me_jd', dt.datetime(2019, 1, 1))
        me_jd.add_const('lambda', 0.3)
        me_jd.add_const('mu', 0.75)
        me_jd.add_const('delta', 0.1)
        #
        me_gbm = MarketEnvironment('me_gbm', dt.datetime(2019, 1, 1))
        me_gbm.add_const('initial_value', 36.0)
        me_gbm.add_const('volatility', 0.2)
        me_gbm.add_const('final_date', dt.datetime(2019, 12, 31))
        me_gbm.add_const('currency', 'RMB')
        me_gbm.add_const('frequency', 'M')
        me_gbm.add_const('paths', 10000)
        csr = ConstShortRate('csr', 0.05)
        me_gbm.add_curve('discount_curve', csr)
        #
        me_jd.add_environment(me_gbm)
        #
        jd = JumpDiffusion('jd', me_jd)
        jd.generate_paths()
        paths = jd.get_instrument_values()
        print(paths)
        # 跳跃概率更大
        jd.update(lamb=0.9)
        jd.generate_paths()
        paths2 = jd.get_instrument_values()
        print(paths2)
        # 绘制图像
        plt.figure(figsize=(8, 4))
        p1 = plt.plot(jd.time_grid, paths[:, :10], 'b')
        p2 = plt.plot(jd.time_grid, paths2[:, :10], 'r-.')
        l1 = plt.legend([p1[0], p2[0]], ['low intensity', 'high intensity'], loc=3)
        plt.gca().add_artist(l1)
        plt.xticks(rotation=30)
        plt.show()

    def exp006(self):
        me_srd = MarketEnvironment('me_srd', dt.datetime(2019, 1, 1))
        me_srd.add_const('initial_value', 0.25)
        me_srd.add_const('volatility', 0.05)
        me_srd.add_const('final_date', dt.datetime(2019, 12, 31))
        me_srd.add_const('currency', 'RMB')
        me_srd.add_const('frequency', 'M')
        me_srd.add_const('paths', 10000)
        me_srd.add_const('kappa', 4.0)
        me_srd.add_const('theta', 0.2)
        me_srd.add_curve('discount_curve', ConstShortRate('r', 0.0))
        srd = SquareRootDiffusion('srd', me_srd)
        paths1 = srd.get_instrument_values()
        print(paths1)
        plt.figure(figsize=(8, 4))
        plt.plot(srd.time_grid, srd.get_instrument_values()[:, :10])
        plt.axhline(me_srd.get_const('theta'), color='r', ls='-', lw=2.0)
        plt.grid(True)
        plt.xticks(rotation=30)
        plt.show()

    def exp007(self):
        me_gbm = MarketEnvironment('gbm', dt.datetime(2019, 1, 1))
        me_gbm.add_const('initial_value', 36.0) # 初始标的价格
        me_gbm.add_const('volatility', 0.2)
        me_gbm.add_const('final_date', dt.datetime(2019, 12, 31))

    def startup(self, args={}):
        #self.exp001()
        #self.exp002()
        #self.exp003()
        #self.exp004()
        #self.exp005()
        #self.exp006()
        #self.exp007()
        va = VolatilityArbitrage()
        va.startup()