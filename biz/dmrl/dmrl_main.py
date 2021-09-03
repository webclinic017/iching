#
import numpy as np
import torch
from biz.dmrl.maml_app import MamlApp
from biz.dmrl.aks_env import AksEnv
from biz.dmrl.aks_util import AksUtil
from biz.dmrl.iqtt.iqtt_app import IqttApp

class DmrlMain(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print("元强化学习量化交易系统 v0.0.1")
        mode = 3 # 1-模型训练；2-模型快速学习；3-模型运行
        app = MamlApp()
        if 1 == mode:
            app.startup(mode=MamlApp.R_M_TRAIN)
        elif 2 == mode:
            app.startup(mode=MamlApp.R_M_RUN_ADAPT_PROCESS)
        elif 3 == mode:
            self.run_model(app=app)
        elif 4 == mode:
            app = IqttApp()
            app.startup()

    def run_model(self, app):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = app.reset()
        model.eval()
        stock_symbol = 'sh600260'
        env = AksEnv(stock_symbol)
        obs = env.reset()
        done = False
        action = env.action_space.sample()
        while not done:
            ## quotation_type = obs[-1].item() #app.run_step(model, torch.from_numpy(np.array([obs[0][:50]])).float().to(device))
            quotation_type = app.run_step(model, torch.from_numpy(np.array([obs[0][:50]])).float().to(device))
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