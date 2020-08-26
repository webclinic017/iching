# 强化学习中的Agent，负责与环境的交互，同时调用对应的策略，生成适合的
# 行动
import numpy as np

class SopAgent(object):
    IDX_OPTION = 0
    IDX_ACTION = 1
    IDX_PERCENT = 2

    def __init__(self):
        self.refl = 'apps.sop.Agent'
        #self.reset(env)

    def reset(self, env):
        self.action = [
            np.zeros((len(env.ds.key_list),)),
            np.zeros((3,)),
            np.zeros((10,))
        ]

    def choose_action(self, obs, reward):
        '''
        根据环境当前状态选择本时间点的行动，将上一时间点行动的奖励信号
        用于策略学习
        '''
        print('看到：{0};\n奖励：{1};'.format(obs, reward))
        self.action[SopAgent.IDX_OPTION][self.action[SopAgent.IDX_OPTION] > 0] = 0
        self.action[SopAgent.IDX_ACTION][self.action[SopAgent.IDX_ACTION] > 0] = 0
        self.action[SopAgent.IDX_PERCENT][self.action[SopAgent.IDX_PERCENT] > 0] = 0
        option_idx = 2
        action_idx = 1
        percent_idx = 2
        self.action[SopAgent.IDX_OPTION][option_idx] = 1
        self.action[SopAgent.IDX_ACTION][action_idx] = 1
        self.action[SopAgent.IDX_PERCENT][percent_idx] = 1
        return self.action