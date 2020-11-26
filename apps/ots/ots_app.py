#
import datetime as dt
from apps.ots.dt_util import DtUtil
from apps.ots.nc_util import NcUtil
from apps.ots.const_short_rate import ConstShortRate
from apps.ots.market_environment import MarketEnvironment

class OtsApp(object):
    def __init__(self):
        self.refl = ''

    def startup(self, args={}):
        #self.exp001()
        #self.exp002()
        self.exp003()

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