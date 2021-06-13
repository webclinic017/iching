# Deep Reinforcement Learning Trader v0.0.1
import pathlib
import numpy as np
import torch
import torch.optim as optim
import gym
from ignite.engine import Engine
from ignite.contrib.handlers import tensorboard_logger as tb_logger
#
import biz.drlt.rll as rll
from biz.drlt.app_config import AppConfig
from biz.drlt.ds.bar_data import BarData
from biz.drlt.envs.minute_bar_env import MinuteBarEnv
from biz.drlt.nns.simple_ff_dqn import SimpleFFDQN
from biz.drlt.nns.dqn_common import DqnCommon
from biz.drlt.nns.dqn_validation import DqnValidation

class DrltApp(object):
    def __init__(self):
        self.name = 'biz.drlt.drlt_app.DrltApp'

    def startup(self, args={}):
        print('深度强化学习交易系统 v0.0.2')
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

        
        def process_batch(engine, batch):
            optimizer.zero_grad()
            loss_v = DqnCommon.calc_loss(
                batch, net, tgt_net.target_model,
                gamma=AppConfig.GAMMA ** AppConfig.REWARD_STEPS, device=device)
            loss_v.backward()
            optimizer.step()
            eps_tracker.frame(engine.state.iteration)

            if getattr(engine.state, "eval_states", None) is None:
                eval_states = buffer.sample(AppConfig.STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False)
                            for transition in eval_states]
                engine.state.eval_states = np.array(eval_states, copy=False)

            return {
                "loss": loss_v.item(),
                "epsilon": selector.epsilon,
            }
        
        engine = Engine(process_batch)
        tb = DqnCommon.setup_ignite(engine, exp_source, f"simple-{run_name}",
                             extra_metrics=('values_mean',))

        @engine.on(rll.ignite.PeriodEvents.ITERS_1000_COMPLETED)
        def sync_eval(engine: Engine):
            tgt_net.sync()

            mean_val = DqnCommon.calc_values_of_states(
                engine.state.eval_states, net, device=device)
            engine.state.metrics["values_mean"] = mean_val
            if getattr(engine.state, "best_mean_val", None) is None:
                engine.state.best_mean_val = mean_val
            if engine.state.best_mean_val < mean_val:
                print("%d: Best mean value updated %.3f -> %.3f" % (
                    engine.state.iteration, engine.state.best_mean_val,
                    mean_val))
                path = saves_path / ("mean_value-%.3f.data" % mean_val)
                torch.save(net.state_dict(), path)
                engine.state.best_mean_val = mean_val

        @engine.on(rll.ignite.PeriodEvents.ITERS_10000_COMPLETED)
        def validate(engine: Engine):
            res = DqnValidation.validation_run(env_tst, net, device=device)
            print("%d: tst: %s" % (engine.state.iteration, res))
            for key, val in res.items():
                engine.state.metrics[key + "_tst"] = val
            res = DqnValidation.validation_run(env_val, net, device=device)
            print("%d: val: %s" % (engine.state.iteration, res))
            for key, val in res.items():
                engine.state.metrics[key + "_val"] = val
            val_reward = res['episode_reward']
            if getattr(engine.state, "best_val_reward", None) is None:
                engine.state.best_val_reward = val_reward
            if engine.state.best_val_reward < val_reward:
                print("Best validation reward updated: %.3f -> %.3f, model saved" % (
                    engine.state.best_val_reward, val_reward
                ))
                engine.state.best_val_reward = val_reward
                path = saves_path / ("val_reward-%.3f.data" % val_reward)
                torch.save(net.state_dict(), path)

        event = rll.ignite.PeriodEvents.ITERS_10000_COMPLETED
        tst_metrics = [m + "_tst" for m in DqnValidation.METRICS]
        tst_handler = tb_logger.OutputHandler(
            tag="test", metric_names=tst_metrics)
        tb.attach(engine, log_handler=tst_handler, event_name=event)

        val_metrics = [m + "_val" for m in DqnValidation.METRICS]
        val_handler = tb_logger.OutputHandler(
            tag="validation", metric_names=val_metrics)
        tb.attach(engine, log_handler=val_handler, event_name=event)
        engine.run(DqnCommon.batch_generator(buffer, AppConfig.REPLAY_INITIAL, AppConfig.BATCH_SIZE))