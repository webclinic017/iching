# 
import os
import json
import pathlib
import yaml
import torch
from tqdm import trange
import gym
# 
from biz.drlt.ds.bar_data import BarData
from biz.drlt.envs.minute_bar_env import MinuteBarEnv
import ann.dmrl.utils.helpers as iduh # import get_policy_for_env
from ann.dmrl.baseline import LinearFeatureBaseline
from ann.dmrl.samplers.multi_task_sampler import MultiTaskSampler
from ann.dmrl.metalearners import MAMLTRPO
import ann.dmrl.utils.reinforcement_learning as durl # import get_returns

class MamlTrpoEngine(object):
    def __init__(self):
        self.name = 'ann.dmrl.maml_trpo_engine.MamlTrpoEngine'

    def startup(self, args={}):
        print('基于TRPO的元强化学习算法 v0.0.8')
        self.train(args=args)

    def train(self, args={}):
        # 读入配置文件
        with open(args['config'], 'r', encoding='utf-8') as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
        config['device'] = ('cuda' if (torch.cuda.is_available()
                   and config['use_cuda']) else 'cpu')
        # 处理输出目录
        if config['output_folder'] is not None:
            if not os.path.exists(config['output_folder']):
                os.makedirs(config['output_folder'])
            policy_filename = os.path.join(config['output_folder'], 'policy.th')
            config_filename = os.path.join(config['output_folder'], 'config.json')
            with open(config_filename, 'w') as fd:
                #config.update(vars(args))
                json.dump(config, fd, indent=4)
        # 处理Seed可重复性
        if config['seed'] is not None:
            torch.manual_seed(config['seed'])
            torch.cuda.manual_seed_all(config['seed'])
        #env = gym.make(config['env-name'], **config.get('env-kwargs', {}))
        env, env_val, env_tst = self.make_env(config)
        env.close()
        # Policy
        policy = iduh.get_policy_for_env(env,
                                    hidden_sizes=config['hidden-sizes'],
                                    nonlinearity=config['nonlinearity'])
        policy.share_memory()
        # Baseline
        baseline = LinearFeatureBaseline(iduh.get_input_size(env))
        # Sampler
        sampler = MultiTaskSampler(config['env-name'],
                               env_kwargs={},
                               batch_size=config['batch_size'],
                               policy=policy,
                               baseline=baseline,
                               env=env,
                               seed=config['seed'],
                               num_workers=config['num_workers'])
        metalearner = MAMLTRPO(policy,
                           fast_lr=config['fast-lr'],
                           first_order=config['first-order'],
                           device=config['device'])
        num_iterations = 0
        for batch in trange(config['num-batches']):
            print(batch)
            tasks = sampler.sample_tasks(num_tasks=config['meta-batch-size'])
            futures = sampler.sample_async(tasks,
                                       num_steps=config['num-steps'],
                                       fast_lr=config['fast-lr'],
                                       gamma=config['gamma'],
                                       gae_lambda=config['gae-lambda'],
                                       device=config['device'])
            print('MamlTrpoEngine.train before calling step')
            logs = metalearner.step(*futures,
                                    max_kl=config['max-kl'],
                                    cg_iters=config['cg-iters'],
                                    cg_damping=config['cg-damping'],
                                    ls_max_steps=config['ls-max-steps'],
                                    ls_backtrack_ratio=config['ls-backtrack-ratio'])
            print('MamlTrpoEngine.train after calling step')
            train_episodes, valid_episodes = sampler.sample_wait(futures)
            num_iterations += sum(sum(episode.lengths) for episode in train_episodes[0])
            num_iterations += sum(sum(episode.lengths) for episode in valid_episodes)
            logs.update(tasks=tasks,
                        num_iterations=num_iterations,
                        train_returns=durl.get_returns(train_episodes[0]),
                        valid_returns=durl.get_returns(valid_episodes))
        # Save policy
        if config['output_folder'] is not None:
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)
        print('^_^ The End ^_^')

    def make_env(self, config):
        '''
        创建股票的强化学习环境
        '''
        run_name = "yt1"
        saves_path = pathlib.Path(config['saves_dir']) / f"simple-{run_name}"
        saves_path.mkdir(parents=True, exist_ok=True)

        data_path = pathlib.Path(config['stocks'])
        val_path = pathlib.Path(config['val_stocks'])
        year = 2016

        if year is not None or data_path.is_file():
            if year is not None:
                print('load stock data...')
                stock_data = BarData.load_year_data(year)
            else:
                stock_data = {"YNDX": BarData.load_relative(data_path)}
            env = MinuteBarEnv(
                stock_data, bars_count=config['bars_count'], volumes=True)
            env_tst = MinuteBarEnv(
                stock_data, bars_count=config['bars_count'], volumes=True)
        elif data_path.is_dir():
            env = MinuteBarEnv.from_dir(
                data_path, bars_count=config['bars_count'])
            env_tst = MinuteBarEnv.from_dir(
                data_path, bars_count=config['bars_count'])
        else:
            raise RuntimeError("No data to train on")

        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        val_data = {"YNDX": BarData.load_relative(val_path)}
        env_val = MinuteBarEnv(val_data, bars_count=config['bars_count'], volumes=True)
        return env, env_val, env_tst
        #return rll.common.wrappers.wrap_dqn(gym.make(AppConfig.a3c_config['ENV_NAME']))
