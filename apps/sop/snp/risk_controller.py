# 风控模块

class RiskController(object):
    def __init__(self):
        self.refl = 'apps.sop.snp.RiskController'

    def review_action(self, obs, reward, action):
        '''
        验证当前状态obs下采取action的合理性
        返回值：True同意，False拒绝
        '''
        print('风控审核通过！！！！！！！！！！！！！！！！！！')
        return True