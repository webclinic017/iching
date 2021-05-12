#
from fak.option.bs_model import BsModel

class BfwC1e1(object):
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
        Ct = BsModel.calculate_C(St, K, t, T, r, sigma)
        print('v0.0.1 Ct={0};'.format(Ct))
        vega = BsModel.calculate_vega(St, K, t, T, r, sigma)
        print('vega={0};'.format(vega))

        sigma_t = 1200
        for n in range(10000):
            C_t = BsModel.calculate_C(St, K, t, T, r, sigma_t)
            vega_t = BsModel.calculate_vega(St, K, t, T, r, sigma_t)
            sigma_t = sigma_t - 1/vega * (C_t - Ct)
            print('C0={0}; vega={1}; sigma={2};'.format(C_t, vega_t, sigma_t))