#
import numpy as np
import torch
import torch.nn as nn
import warnings
from typing import Iterable
from datetime import datetime, timedelta
# 
import biz.drlt.rll as rll

class DqnCommon(object):
    @staticmethod
    @torch.no_grad()
    def calc_values_of_states(states, net, device="cpu"):
        mean_vals = []
        for batch in np.array_split(states, 64):
            states_v = torch.tensor(batch).to(device)
            action_values_v = net(states_v)
            best_action_values_v = action_values_v.max(1)[0]
            mean_vals.append(best_action_values_v.mean().item())
        return np.mean(mean_vals)

    @staticmethod
    def unpack_batch(batch):
        states, actions, rewards, dones, last_states = [], [], [], [], []
        for exp in batch:
            state = np.array(exp.state, copy=False)
            states.append(state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            if exp.last_state is None:
                last_states.append(state)       # the result will be masked anyway
            else:
                last_states.append(np.array(exp.last_state, copy=False))
        return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)

    @staticmethod
    def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
        states, actions, rewards, dones, next_states = DqnCommon.unpack_batch(batch)

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
        next_state_values[done_mask] = 0.0

        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    @staticmethod
    def batch_generator(buffer: rll.experience.ExperienceReplayBuffer,
                        initial: int, batch_size: int):
        buffer.populate(initial)
        while True:
            buffer.populate(1)
            yield buffer.sample(batch_size)