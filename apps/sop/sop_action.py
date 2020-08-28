# 期权交易行动类
import numpy as np

class SopAction(object):
    IDX_OPTION = 0
    IDX_ACTION = 1
    IDX_PERCENT = 2

    def __init__(self, env):
        self.refl = 'apps.sop.SopAction'
        self.action = [
            np.zeros((len(env.ds.key_list),)),
            np.zeros((3,)),
            np.zeros((10,))
        ]

    def reset(self):
        self.action[SopAction.IDX_OPTION][self.action\
                    [SopAction.IDX_OPTION] > 0] = 0
        self.action[SopAction.IDX_ACTION][self.action\
                    [SopAction.IDX_ACTION] > 0] = 0
        self.action[SopAction.IDX_PERCENT][self.action\
                    [SopAction.IDX_PERCENT] > 0] = 0
        return self.action