# 策略类基类
from apps.sop.sop_registry import SopRegistry
import sys
from apps.sop.sop_action import SopAction

class BaseStrategy(object):
    def __init__(self, action):
        self.refl = 'apps.sop.snp.BaseStragegy'
        self.action = action

    def run(self, obs, reward):
        print('策略类看到的环境状态：{0};'.format(obs['X'].shape))
        print('期权编号：{0};'.format(SopRegistry.get(SopRegistry.K_OPTION_CODES)))
        # 获取期权行情数据
        sys.exit(0)
        self.action.reset()
        option_idx = 0
        self.action.action[SopAction.IDX_OPTION][option_idx] = 1
        action_idx = 0
        self.action.action[SopAction.IDX_ACTION][action_idx] = 1
        percent_idx = 9
        self.action.action[SopAction.IDX_PERCENT][percent_idx] = 1
        return self.action.action

    def reset(self):
        self.action.reset()