#
import numpy as np
from apps.ots.valuation.valuation_base import ValuationBase

class ValuationMcsEu(ValuationBase):
    ''' 对任意支付的欧式期权进行定价（用蒙特卡洛模拟方法） '''
    def generate_payoff(self, fixed_seed=False):
        ''' 
        只要规定期权价值和标的资产价格的一个关系，我们用蒙特卡洛模拟的方法算出期权的价值
        给定标的价格，得到期权到期日价格（采用蒙特卡洛模拟方法）
        '''
        strike = None
        if self.strike is not None:
            strike = self.strike # 行权价格
        paths = self.underlying.get_instrument_values(fixed_seed=fixed_seed)
        time_grid = self.underlying.time_grid
        time_index = 0
        try:
            time_index = np.where(time_grid == self.maturity)[0]
            time_index = int(time_index)
        except:
            print("到期日不在时间范围内")
        # 标的在到期日价格、平均价格、最大价格、最小价格
        maturity_value = paths[time_index] 
        mean_value = np.mean(paths[:time_index], axis=1)
        max_value = np.amax(paths[:time_index], axis=1)[-1]
        min_value = np.amin(paths[:time_index], axis=1)[-1]
        try:
            payoff = eval(payoff_func)
            return payoff
        except:
            print('求到期日价格时出错')

    def present_value(self, accuracy=6, fixed_seed=False, full=False):
        cash_flow = self.generate_payoff(fixed_seed=fixed_seed)
        discount_factor = self.discount_curve.get_discount_factors(
                    (self.pricing_date, self.maturity))[0, 1] # 获取贴现因子
        result = discount_factor * np.sum(cash_flow) / len(cash_flow)
        if full:
            return round(result, accuracy), discount_factor * cash_flow
        else:
            return round(result, accuracy)