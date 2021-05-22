# A3C算法示例
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn.utils as nn_utils
import gym
#
from ann.iching_writer import IchingWriter
import biz.drlt.rll as rll
from apps.drl.c13.app_config import AppConfig
from biz.drlt.nns.a2c_model import A2cModel
from biz.drlt.nns.a3c_common import A3cCommon
from biz.drlt.nns.reward_tracker import RewardTracker

class C13App(object):
    def __init__(self):
        self.name = 'apps.drl.c13.c13_app.C13App'

    def startup(self):
        print('A3C算法（梯度集成）示例')

        mp.set_start_method('spawn')
        os.environ['OMP_NUM_THREADS'] = "1"
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument("--cuda", default=False,
                            action="store_true", help="Enable cuda")
        parser.add_argument("-n", "--name", required=True,
                            help="Name of the run")
        args = parser.parse_args()
        '''
        device = 'cuda:0'
        run_name = 'a3c'

        env = C13App.make_env()
        net = A2cModel(env.observation_space.shape,
                            env.action_space.n) #.to(device)
        for param in net.parameters():
            print('##### pos 0.01 #####  dthread grads: {0};'.format(type(param.grad)))

        net.share_memory()

        for param in net.parameters():
            print('##### pos 0.02 #####  dthread grads: {0};'.format(type(param.grad)))

        optimizer = optim.Adam(net.parameters(),
                            lr=AppConfig.LEARNING_RATE, eps=1e-3)

        train_queue = mp.Queue(maxsize=AppConfig.PROCESSES_COUNT)
        data_proc_list = []
        for proc_idx in range(AppConfig.PROCESSES_COUNT):
            proc_name = f"-a3c-grad_pong_{run_name}#{proc_idx}"
            p_args = (proc_name, net, device, train_queue)
            data_proc = mp.Process(target=C13App.grads_func, args=p_args)
            data_proc.start()
            data_proc_list.append(data_proc)

        batch = []
        step_idx = 0
        grad_buffer = None

        for param in net.parameters():
            print('##### pos1 #####  dthread grads: {0};'.format(type(param.grad)))

        try:
            while True:
                train_entry = train_queue.get()
                if train_entry is None:
                    break
                step_idx += 1
                if grad_buffer is None:
                    grad_buffer = train_entry
                else:
                    for tgt_grad, grad in zip(grad_buffer,
                                            train_entry):
                        tgt_grad += grad
                if step_idx % AppConfig.TRAIN_BATCH == 0:
                    net.zero_grad() #yt

                    
                    for param in net.parameters():
                        print('##### pos200 #####  dthread grads: {0};'.format(type(param.grad)))



                    for param, grad in zip(net.parameters(),
                                        grad_buffer):
                        v1 = torch.FloatTensor(grad).to(device)
                        print('param.grad: {0}; grad: {1}; device:{2}; param: {3};'.format(param.grad, v1.shape, device, param.shape))
                        if param.grad is not None:
                            param.grad = torch.FloatTensor(grad).to(device)

                    nn_utils.clip_grad_norm_(
                        net.parameters(), AppConfig.CLIP_GRAD)
                    optimizer.step()
                    grad_buffer = None
        finally:
            for p in data_proc_list:
                p.terminate()
                p.join()




    @staticmethod
    def make_env():
        return rll.common.wrappers.wrap_dqn(gym.make(AppConfig.ENV_NAME))

    @staticmethod
    def grads_func(proc_name, net, device, train_queue):
        net.to(device)
        envs = [C13App.make_env() for _ in range(AppConfig.NUM_ENVS)]

        agent = rll.agent.PolicyAgent(
            lambda x: net(x)[0], device=device, apply_softmax=True)
        exp_source = rll.experience.ExperienceSourceFirstLast(
            envs, agent, gamma=AppConfig.GAMMA, steps_count=AppConfig.REWARD_STEPS)

        batch = []
        frame_idx = 0

        writer = IchingWriter()
        with RewardTracker(writer, AppConfig.REWARD_BOUND) as tracker:
            with rll.common.utils.TBMeanTracker(
                    writer, 100) as tb_tracker:
                for exp in exp_source:
                    frame_idx += 1
                    new_rewards = exp_source.pop_total_rewards()
                    if new_rewards and tracker.reward(
                            new_rewards[0], frame_idx):
                        break

                    batch.append(exp)
                    if len(batch) < AppConfig.GRAD_BATCH:
                        continue

                    data = A3cCommon.unpack_batch(
                        batch, net, device=device,
                        last_val_gamma=AppConfig.GAMMA**AppConfig.REWARD_STEPS)
                    states_v, actions_t, vals_ref_v = data

                    batch.clear()

                    net.zero_grad()
                    logits_v, value_v = net(states_v)
                    loss_value_v = F.mse_loss(
                        value_v.squeeze(-1), vals_ref_v)

                    log_prob_v = F.log_softmax(logits_v, dim=1)
                    adv_v = vals_ref_v - value_v.detach()
                    log_p_a = log_prob_v[range(AppConfig.GRAD_BATCH), actions_t]
                    log_prob_actions_v = adv_v * log_p_a
                    loss_policy_v = -log_prob_actions_v.mean()

                    prob_v = F.softmax(logits_v, dim=1)
                    ent = (prob_v * log_prob_v).sum(dim=1).mean()
                    entropy_loss_v = AppConfig.ENTROPY_BETA * ent

                    loss_v = entropy_loss_v + loss_value_v + \
                            loss_policy_v
                    loss_v.backward()

                    tb_tracker.track("advantage", adv_v, frame_idx)
                    tb_tracker.track("values", value_v, frame_idx)
                    tb_tracker.track("batch_rewards", vals_ref_v,
                                    frame_idx)
                    tb_tracker.track("loss_entropy", entropy_loss_v,
                                    frame_idx)
                    tb_tracker.track("loss_policy", loss_policy_v,
                                    frame_idx)
                    tb_tracker.track("loss_value", loss_value_v,
                                    frame_idx)
                    tb_tracker.track("loss_total", loss_v, frame_idx)

                    # gather gradients
                    nn_utils.clip_grad_norm_(
                        net.parameters(), AppConfig.CLIP_GRAD)
                    grads = [
                        param.grad.data.cpu().numpy()
                        if param.grad is not None else None
                        for param in net.parameters()
                    ]
                    train_queue.put(grads)
                    
                    for param in net.parameters():
                        print('##### pos AAA #####  dthread grads: {0};'.format(type(param.grad)))

        train_queue.put(None)
