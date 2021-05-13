#
import math
import unittest
from fak.option.bsm_model import BsmModel

class TBsmModel(unittest.TestCase):
    def test_main(self):
        # 初始化变量
        St = 2115.0 # 标的资产价格
        K = 2100.0 # 行权价格
        t = 10 # 当前时间
        T = 10.0833 # 到期时间
        r = 0.05 # 短期无风险利率
        sigma = 0.1676 # 波动率
        # 假设已知波动率的情况下计算期权价格和vega
        C = BsmModel.calculate_C(St, K, t, T, r, sigma)
        print('C={0};'.format(C))
        vega = BsmModel.calculate_vega(St, K, t, T, r, sigma)
        # 随机给出一个波动率，用迭代法求出波动率真值
        sigma_t = 1200
        for n in range(10000):
            C_t = BsmModel.calculate_C(St, K, t, T, r, sigma_t)
            vega_t = BsmModel.calculate_vega(St, K, t, T, r, sigma_t)
            sigma_t = sigma_t - 1/vega * (C_t - C)
            print('C0={0}; vega={1}; sigma={2};'.format(C_t, vega_t, sigma_t))
        print('真实值：C={0}; vega={1};'.format(C, vega))
        rst = math.fabs(C - C_t)<0.1 and math.fabs(sigma - sigma_t)<0.1
        self.assertTrue(rst)