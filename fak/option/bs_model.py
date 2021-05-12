#
import math
from enum import Enum,unique
from scipy.stats import norm

@unique
class DMode(Enum):
    D1 = 1
    D2 = 2

class BsModel(object):
    def __init__(self):
        self.name = 'fak.option.bs_model.BsModel'
    
    @staticmethod
    def calculate_C(St, K, t, T, r, sigma):
        '''
        已知波动率计算期权价格的BS公式
        '''
        mu = 0.0
        sigma = 1.0
        d1 = BsModel.calculate_d1(St, K, t, T, r, sigma)
        d2 = BsModel.calculate_d2(St, K, t, T, r, sigma)
        nd1 = BsModel.calculate_nd(d1, mu, sigma)
        nd2 = BsModel.calculate_nd(d2, mu, sigma)
        return St * nd1 - math.exp(-r*(T-t)) * K * nd2

    @staticmethod
    def calculate_vega(St, K, t, T, r, sigma):
        '''
        计算期权价格对d1的导数，即希腊字母vega的值
        '''
        d1 = BsModel.calculate_d1(St, K, t, T, r, sigma)
        nd1p = BsModel.calculate_ndp(d1)
        return St * nd1p * math.sqrt(T-t)

    @staticmethod
    def calculate_d1(St, K, t, T, r, sigma):
        return BsModel.calculate_d(St, K, t, T, r, sigma, DMode.D1)

    @staticmethod
    def calculate_d2(St, K, t, T, r, sigma):
        return BsModel.calculate_d(St, K, t, T, r, sigma, DMode.D2)

    @staticmethod
    def calculate_d(St, K, t, T, r, sigma, mode):
        if DMode.D1 == mode:
            numerator = math.log(St / K)  + (r + sigma*sigma / 2.0)*(T - t)
        else:
            numerator = math.log(St / K) + (r - sigma*sigma / 2.0)*(T - t)
        denominator = sigma * math.sqrt(T-t)
        return numerator / denominator
        
    @staticmethod
    def calculate_nd(x0, mu=0.0, sigma=1.0):
        '''
        计算由均值为mu，方差为sigma的正态分布下，x<=x0时的概率
        '''
        return norm.cdf(x0, loc=mu, scale=sigma)

    @staticmethod
    def calculate_ndp(x0, mu=0.0, sigma=1.0):
        return norm.pdf(x0, loc=mu, scale=sigma)