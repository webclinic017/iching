# 
import pathlib
import numpy as np
import torch
import torch.optim as optim
import gym
#
import biz.drlt.rll as rll
from biz.drlt.app_config import AppConfig
from biz.drlt.ds.bar_data import BarData
from biz.drlt.envs.minute_bar_env import MinuteBarEnv
from biz.drlt.nns.simple_ff_dqn import SimpleFFDQN
from biz.drlt.nns.dqn_common import DqnCommon
from biz.drlt.nns.dqn_validation import DqnValidation

class DqnApp(object):
    def __init__(self):
        self.name = 'biz.drlt.dqn_app.DqnApp'

    def train(self):
        print('DQN股票交易系统 v0.0.1')
        device = torch.device("cuda:0")
        #device = torch.device("cpu")
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
                stock_data, bars_count=AppConfig.BARS_COUNT)
            env_tst = MinuteBarEnv(
                stock_data, bars_count=AppConfig.BARS_COUNT)
        elif data_path.is_dir():
            env = MinuteBarEnv.from_dir(
                data_path, bars_count=AppConfig.BARS_COUNT)
            env_tst = MinuteBarEnv.from_dir(
                data_path, bars_count=AppConfig.BARS_COUNT)
        else:
            raise RuntimeError("No data to train on")

        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        val_data = {"YNDX": BarData.load_relative(val_path)}
        env_val = MinuteBarEnv(val_data, bars_count=AppConfig.BARS_COUNT)

        net = SimpleFFDQN(env.observation_space.shape[0],
                                env.action_space.n).to(device)
        tgt_net = rll.agent.TargetNet(net)

        selector = rll.actions.EpsilonGreedyActionSelector(AppConfig.EPS_START)
        eps_tracker = rll.actions.EpsilonTracker(
            selector, AppConfig.EPS_START, AppConfig.EPS_FINAL, AppConfig.EPS_STEPS)
        agent = rll.agent.DQNAgent(net, selector, device=device)
        exp_source = rll.experience.ExperienceSourceFirstLast(
            env, agent, AppConfig.GAMMA, steps_count=AppConfig.REWARD_STEPS)
        buffer = rll.experience.ExperienceReplayBuffer(
            exp_source, AppConfig.REPLAY_SIZE)
        optimizer = optim.Adam(net.parameters(), lr=AppConfig.LEARNING_RATE)

        
        epochs = 1000
        sync_tgt_net_per_iters = 1000
        validate_per_iters = 10000
        metrics = {}
        iter_num = 0
        best_mean_val = None
        best_val_reward = None
        eval_states = None
        for epoch in range(epochs):
            for batch in DqnCommon.batch_generator(buffer, AppConfig.REPLAY_INITIAL, AppConfig.BATCH_SIZE):
                optimizer.zero_grad()
                loss_v = DqnCommon.calc_loss(
                    batch, net, tgt_net.target_model,
                    gamma=AppConfig.GAMMA ** AppConfig.REWARD_STEPS, device=device)
                loss_v.backward()
                optimizer.step()
                eps_tracker.frame(iter_num) # engine.state.iteration)
                iter_num += 1
                if iter_num % 100 == 0:
                    print('epoch_{0}_{1}: loss={2};'.format(epoch, iter_num, loss_v))
                if eval_states is None:
                    eval_states = buffer.sample(AppConfig.STATES_TO_EVALUATE)
                    eval_states = [np.array(transition.state, copy=False)
                            for transition in eval_states]
                    eval_states = np.array(eval_states, copy=False)
                # 隔sync_tgt_net_per_iters次更新target network参数
                if iter_num % sync_tgt_net_per_iters == 0:
                    print('synchronize target network with working network')
                    tgt_net.sync()
                    mean_val = DqnCommon.calc_values_of_states(
                        eval_states, net, device=device)
                    metrics["values_mean"] = mean_val
                    if best_mean_val is None:
                        best_mean_val = mean_val
                    if best_mean_val < mean_val:
                        print("%d: Best mean value updated %.3f -> %.3f" % (
                            iter_num, best_mean_val,
                            mean_val))
                        path = saves_path / ("mean_value-%.3f.data" % mean_val)
                        torch.save(net.state_dict(), path)
                        best_mean_val = mean_val
                # 每validate_per_iters次运行一遍验证集
                if iter_num % validate_per_iters == 0:
                    res = DqnValidation.validation_run(env_tst, net, device=device)
                    print("%d: tst: %s" % (iter_num, res))
                    for key, val in res.items():
                        metrics[key + "_tst"] = val
                    res = DqnValidation.validation_run(env_val, net, device=device)
                    print("%d: val: %s" % (iter_num, res))
                    for key, val in res.items():
                        metrics[key + "_val"] = val
                    val_reward = res['episode_reward']
                    if best_val_reward is None:
                        best_val_reward = val_reward
                    if best_val_reward < val_reward:
                        print("Best validation reward updated: %.3f -> %.3f, model saved" % (
                            best_val_reward, val_reward
                        ))
                        best_val_reward = val_reward
                        path = saves_path / ("val_reward-%.3f.data" % val_reward)
                        torch.save(net.state_dict(), path)