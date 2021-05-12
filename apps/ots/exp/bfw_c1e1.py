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