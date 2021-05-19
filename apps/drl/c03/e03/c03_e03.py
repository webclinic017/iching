#
from apps.drl.c03.e03.sin_env import SinEnv

class C03E03(object):
    def __init__(self):
        self.name = 'apps.drl.c03.e03.c03_e03.C03E03'

    def startup(self, args={}):
        print('深度强化学习环境与动画绘制集成方案')
        env = SinEnv()
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                env.close()
                break
            env.render()