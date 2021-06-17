# A3C算法示例
import os
import pathlib
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn.utils as nn_utils
import gym
#
from ann.iching_writer import IchingWriter
from biz.drlt.ds.bar_data import BarData
from biz.drlt.envs.minute_bar_env import MinuteBarEnv
import biz.drlt.rll as rll
from biz.drlt.app_config import AppConfig
from biz.drlt.nns.a2c_conv1d_model import A2cConv1dModel
from biz.drlt.nns.a3c_common import A3cCommon
from biz.drlt.nns.reward_tracker import RewardTracker

class A3cApp(object):
    def __init__(self):
        self.name = 'biz.drlt.a3c_app.A3cApp'

    def train(self):
        print('A3C算法股票交易系统 v0.0.0_1(Pong)')

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

        env, env_val, env_tst = A3cApp.make_env()
        #env = A3cApp.make_env()
        print('shape: {0}; n: {1};'.format(env.observation_space.shape, env.action_space.n))
        net = A2cConv1dModel((1, env.observation_space.shape[0]),
                            env.action_space.n) #.to(device)
        net.share_memory()
        optimizer = optim.Adam(net.parameters(),
                            lr=AppConfig.a3c_config['learning_rate'], eps=1e-3)

        train_queue = mp.Queue(maxsize=AppConfig.a3c_config['processes_count'])
        data_proc_list = []
        for proc_idx in range(AppConfig.a3c_config['processes_count']):
            proc_name = f"-a3c-grad_pong_{run_name}#{proc_idx}"
            p_args = (proc_name, net, device, train_queue)
            data_proc = mp.Process(target=A3cApp.grads_func, args=p_args)
            data_proc.start()
            data_proc_list.append(data_proc)

        batch = []
        step_idx = 0
        grad_buffer = None
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
                if step_idx % AppConfig.a3c_config['train_batch'] == 0:
                    net.zero_grad() #yt
                    for param, grad in zip(net.parameters(),
                                        grad_buffer):
                        v1 = torch.FloatTensor(grad).to(device)
                        if param.grad is not None:
                            param.grad = torch.FloatTensor(grad).to(device)

                    nn_utils.clip_grad_norm_(
                        net.parameters(), AppConfig.a3c_config['clip_grad'])
                    optimizer.step()
                    grad_buffer = None
        finally:
            for p in data_proc_list:
                p.terminate()
                p.join()




    @staticmethod
    def make_env():
        run_name = "yt1"
        saves_path = AppConfig.SAVES_DIR / f"simple-{run_name}"
        saves_path.mkdir(parents=True, exist_ok=True)

        data_path = pathlib.Path(AppConfig.STOCKS)
        val_path = pathlib.Path(AppConfig.VAL_STOCKS)
        year = 2016

        if year is not None or data_path.is_file():
            if year is not None:
                print('load stock data...')
                stock_data = BarData.load_year_data(year)
            else:
                stock_data = {"YNDX": BarData.load_relative(data_path)}
            env = MinuteBarEnv(
                stock_data, bars_count=AppConfig.BARS_COUNT, volumes=True)
            env_tst = MinuteBarEnv(
                stock_data, bars_count=AppConfig.BARS_COUNT, volumes=True)
        elif data_path.is_dir():
            env = MinuteBarEnv.from_dir(
                data_path, bars_count=AppConfig.BARS_COUNT)
            env_tst = MinuteBarEnv.from_dir(
                data_path, bars_count=AppConfig.BARS_COUNT)
        else:
            raise RuntimeError("No data to train on")

        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        val_data = {"YNDX": BarData.load_relative(val_path)}
        env_val = MinuteBarEnv(val_data, bars_count=AppConfig.BARS_COUNT, volumes=True)
        return env, env_val, env_tst
        #return rll.common.wrappers.wrap_dqn(gym.make(AppConfig.a3c_config['ENV_NAME']))

    @staticmethod
    def grads_func(proc_name, net, device, train_queue):
        net.to(device)
        #envs = [A3cApp.make_env() for _ in range(AppConfig.a3c_config['NUM_ENVS'])]
        env, env_val, env_tst = A3cApp.make_env()
        envs = [env]

        agent = rll.agent.PolicyAgent(
            lambda x: net(x)[0], device=device, apply_softmax=True)
        exp_source = rll.experience.ExperienceSourceFirstLast(
            envs, agent, gamma=AppConfig.a3c_config['gamma'], steps_count=AppConfig.a3c_config['reward_steps'])

        batch = []
        frame_idx = 0

        writer = IchingWriter()
        with RewardTracker(writer, AppConfig.a3c_config['reward_bound']) as tracker:
            with rll.common.utils.TBMeanTracker(
                    writer, 100) as tb_tracker:
                for exp in exp_source:
                    frame_idx += 1
                    new_rewards = exp_source.pop_total_rewards()
                    if new_rewards and tracker.reward(
                            new_rewards[0], frame_idx):
                        break

                    batch.append(exp)
                    if len(batch) < AppConfig.a3c_config['grad_batch']:
                        continue

                    data = A3cCommon.unpack_batch(
                        batch, net, device=device,
                        last_val_gamma=AppConfig.a3c_config['gamma']**AppConfig.a3c_config['reward_steps'])
                    states_v, actions_t, vals_ref_v = data

                    batch.clear()

                    net.zero_grad()
                    logits_v, value_v = net(states_v)
                    loss_value_v = F.mse_loss(
                        value_v.squeeze(-1), vals_ref_v)

                    log_prob_v = F.log_softmax(logits_v, dim=1)
                    adv_v = vals_ref_v - value_v.detach()
                    log_p_a = log_prob_v[range(AppConfig.a3c_config['grad_batch']), actions_t]
                    log_prob_actions_v = adv_v * log_p_a
                    loss_policy_v = -log_prob_actions_v.mean()

                    prob_v = F.softmax(logits_v, dim=1)
                    ent = (prob_v * log_prob_v).sum(dim=1).mean()
                    entropy_loss_v = AppConfig.a3c_config['entropy_beta'] * ent

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
                        net.parameters(), AppConfig.a3c_config['clip_grad'])
                    grads = [
                        param.grad.data.cpu().numpy()
                        if param.grad is not None else None
                        for param in net.parameters()
                    ]
                    train_queue.put(grads)

        train_queue.put(None)