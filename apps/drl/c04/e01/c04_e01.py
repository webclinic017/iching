#
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
# 
from ann.iching_tensor_board import IchingTensorBoard
from apps.drl.c04.e01.app_config import AppConfig
from apps.drl.c04.e01.cerl_mlp import CerlMlp

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

class C04E01(object):
    def __init__(self):
        self.name = 'apps.drl.c04.e01.c04_e01.C04E01'
        self.itb = IchingTensorBoard()

    def startup(self, args={}):
        print('CrossEntroy Policy Gradient Method')
        env = gym.make("CartPole-v0")
        # env = gym.wrappers.Monitor(env, directory="mon", force=True)
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n

        net = CerlMlp(obs_size, AppConfig.HIDDEN_SIZE, n_actions)
        objective = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=net.parameters(), lr=0.01)

        for iter_no, batch in enumerate(self.iterate_batches(
                env, net, AppConfig.BATCH_SIZE)):
            obs_v, acts_v, reward_b, reward_m = \
                self.filter_batch(batch, AppConfig.PERCENTILE)
            optimizer.zero_grad()
            action_scores_v = net(obs_v)
            loss_v = objective(action_scores_v, acts_v)
            loss_v.backward()
            optimizer.step()
            print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
                iter_no, loss_v.item(), reward_m, reward_b))
            self.itb.update_plot(iter_no, reward_m)
            env.render() # 在界面上显示结果
            if reward_m > 199:
                print("Solved!")
                env.close()
                self.itb.stop_plot()
                break


    def iterate_batches(self, env, net, batch_size):
        batch = []
        episode_reward = 0.0
        episode_steps = []
        obs = env.reset()
        sm = nn.Softmax(dim=1)
        while True:
            obs_v = torch.FloatTensor([obs])
            act_probs_v = sm(net(obs_v))
            act_probs = act_probs_v.data.numpy()[0]
            action = np.random.choice(len(act_probs), p=act_probs)
            next_obs, reward, is_done, _ = env.step(action)
            episode_reward += reward
            step = EpisodeStep(observation=obs, action=action)
            episode_steps.append(step)
            if is_done:
                e = Episode(reward=episode_reward, steps=episode_steps)
                batch.append(e)
                episode_reward = 0.0
                episode_steps = []
                next_obs = env.reset()
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            obs = next_obs


    def filter_batch(self, batch, percentile):
        rewards = list(map(lambda s: s.reward, batch))
        reward_bound = np.percentile(rewards, percentile)
        reward_mean = float(np.mean(rewards))

        train_obs = []
        train_act = []
        for reward, steps in batch:
            if reward < reward_bound:
                continue
            train_obs.extend(map(lambda step: step.observation, steps))
            train_act.extend(map(lambda step: step.action, steps))

        train_obs_v = torch.FloatTensor(train_obs)
        train_act_v = torch.LongTensor(train_act)
        return train_obs_v, train_act_v, reward_bound, reward_mean