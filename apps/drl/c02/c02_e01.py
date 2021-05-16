#
import gym

class C02E01(object):
    def __init__(self):
        self.name = 'apps.drl.c02.c02_e01.C02E01'

    def startup(self, args={}):
        env = gym.make("CartPole-v0")
        total_reward = 0.0
        total_steps = 0
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            total_steps += 1
            if done:
                break
        print("Episode done in %d steps, total reward %.2f" % (
            total_steps, total_reward))