# 策略类基类
from apps.sop.sop_registry import SopRegistry
import sys
from apps.sop.sop_action import SopAction
from apps.sop.sop_registry import SopRegistry

class BaseStrategy(object):
    def __init__(self, action):
        self.refl = 'apps.sop.snp.BaseStragegy'
        self.action = action

    def run(self, obs, reward):
        print('策略类看到的环境状态：{0};'.format(obs['X'].shape))
        option_codes = SopRegistry.get(SopRegistry.K_OPTION_CODES)
        oc = option_codes[0]
        print('oc: {0};'.format(option_codes[0]))
        oc_quotations = []
        asset_quotations = []
        for day_idx in range(len(option_codes)):
            daily_quotation = obs['X'][day_idx*373 : (day_idx+1)*373]
            oc_idx = SopRegistry.get(SopRegistry.K_OC_TO_IDX)[oc]
            oc_quotations += daily_quotation[oc_idx*8 : (oc_idx+1)*8]
            asset_quotations += daily_quotation[-5:]
        print('oc_quotations: {0};'.format(oc_quotations))
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