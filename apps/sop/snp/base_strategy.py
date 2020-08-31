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
        option_codes = SopRegistry.get(SopRegistry.K_OPTION_CODES)
        day_idx = 0
        daily_quotations = obs['X'][day_idx*373 : (day_idx+1)*373]
        oc_idx = 0
        oc_quotation = daily_quotations[oc_idx*8 : (oc_idx+1)*8]
        asset_quotation = daily_quotations[-5 : ]
        print('期权{0}：{1}; 标的行情：{2};'.format(option_codes[day_idx], oc_quotation, asset_quotation))
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