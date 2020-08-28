# 强化学习中的Agent，负责与环境的交互，同时调用对应的策略，生成适合的
# 行动
import numpy as np
#
from apps.sop.snp.base_strategy import BaseStrategy
from apps.sop.snp.risk_controller import RiskController

class SopAgent(object):

    def __init__(self, action):
        self.refl = 'apps.sop.Agent'
        #self.reset(env)
        self.strategy = BaseStrategy(action)
        self.risk_controller = RiskController()

    def reset(self, env):
        self.strategy.reset()

    def choose_action(self, obs, reward):
        '''
        根据环境当前状态选择本时间点的行动，将上一时间点行动的奖励信号
        用于策略学习
        '''
        print('看到：{0};\n奖励：{1};'.format(obs, reward))
        action = self.strategy.run(obs, reward)
        if not self.risk_controller.review_action(obs, reward, action):
            action.reset()
        return action
