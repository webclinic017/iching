#
import numpy as np
import torch
from biz.dmrl.maml_app import MamlApp
from biz.dmrl.aks_env import AksEnv
from biz.dmrl.aks_util import AksUtil

class DmrlMain(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print("元强化学习量化交易系统 v0.0.1")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        app = MamlApp()
        model = app.reset()
        model.eval()
        #app.startup()
        stock_symbol = 'sh600260'
        env = AksEnv(stock_symbol)
        obs = env.reset()
        done = False
        action = env.action_space.sample()
        while not done:
            quotation_type = obs[-1].item() #app.run_step(model, torch.from_numpy(np.array([obs[0][:50]])).float().to(device))
            if quotation_type == 0:
                # 买入
                action[0] = 0.5
                action[1] = 1.0
            elif quotation_type == 1:
                action[0] = 1.5
                action[1] = 1.0
            else:
                action[0] = 2.5
                action[1] = 0.5
            obs, reward, done, info = env.step(action)