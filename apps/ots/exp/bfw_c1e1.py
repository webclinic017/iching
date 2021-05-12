#
import math
from enum import Enum,unique
from scipy.stats import norm

@unique
class DMode(Enum):
    D1 = 1
    D2 = 2

class BfwC1e1(object):
    MODE_D1 = 1
    MODE_D2 = 2

    def __init__(self):
        self.name = ''

    def startup(self):
        print('北风网教程第1章例1')
        St = 2100.0 # 标的资产价格
        K = 2200.0 # 行权价格
        t = 10 # 当前时间
        T = 30 # 到期时间
        r = 0.01 # 短期无风险利率
        sigma = 0.3 # 波动率
        Ct = self.calculate_bsm(St, K, t, T, r, sigma)
        print('Ct={0};'.format(Ct))
        vega = self.calculate_vega(St, K, t, T, r, sigma)
        print('vega={0};'.format(vega))

    def calculate_bsm(self, St, K, t, T, r, sigma):
        mu = 0.0
        sigma = 1.0
        d1 = self.calculate_bsm_d1(St, K, t, T, r, sigma)
        d2 = self.calculate_bsm_d2(St, K, t, T, r, sigma)
        nd1 = self.calculate_bsm_nd(d1, mu, sigma)
        nd2 = self.calculate_bsm_nd(d2, mu, sigma)
        return St * nd1 - math.exp(-r*(T-t)) * K * nd2

    def calculate_vega(self, St, K, t, T, r, sigma):
        d1 = self.calculate_bsm_d1(St, K, t, T, r, sigma)
        nd1p = self.calculate_bsm_ndp(d1)
        return St * nd1p * math.sqrt(T-t)

    def calculate_bsm_d1(self, St, K, t, T, r, sigma):
        return self.calculate_bsm_d(St, K, t, T, r, sigma, DMode.D1)

    def calculate_bsm_d2(self, St, K, t, T, r, sigma):
        return self.calculate_bsm_d(St, K, t, T, r, sigma, DMode.D2)

    def calculate_bsm_d(self, St, K, t, T, r, sigma, mode):
        if DMode.D1 == mode:
            numerator = math.log(St / K)  + (r + sigma*sigma / 2.0)*(T - t)
        else:
            numerator = math.log(St / K) + (r - sigma*sigma / 2.0)*(T - t)
        denominator = sigma * math.sqrt(T-t)
        return numerator / denominator
        

    def calculate_bsm_nd(self, x0, mu=0.0, sigma=1.0):
        '''
        计算由均值为mu，方差为sigma的正态分布下，x<=x0时的概率
        '''
        return norm.cdf(x0, loc=mu, scale=sigma)

    def calculate_bsm_ndp(self, x0, mu=0.0, sigma=1.0):
        return norm.pdf(x0, loc=mu, scale=sigma)