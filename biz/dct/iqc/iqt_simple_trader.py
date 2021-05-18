import numpy as np
#
import ray
from ray import tune
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
#
from iqt.oms.instruments.instrument import Instrument
from iqt.env.default.actions import BSH
from iqt.env.default.rewards import PBR
from biz.dct.iqc.position_change_chart import PositionChangeChart
import iqt.env.default as default
from iqt.feed.core import DataFeed, Stream
from iqt.oms.exchanges import Exchange
from iqt.oms.services.execution.simulated import execute_order
from iqt.oms.wallets import Wallet, Portfolio

class IqtSimpleTrader(object):
    def __init__(self):
        self.name = 'biz.dct.iqc.iqt_simple_trader.IqtSimpleTrader'

    def startup(self, args={}):
        #ray.init(num_cpus=1, num_gpus=0, _redis_password='yantao')
        # 生成交易对象
        self.train()
        #self.evaluate()

    def train(self):
        # Instantiate the environment
        env = self.create_env({
            "window_size": 25
        })
        # Run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()
        num = 0
        while not done:
            # action = agent.compute_action(obs)
            action = self.agent.sample()
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            num += 1
            if num > 5000:
                break
        env.render()

        '''
        register_env("TradingEnv", self.create_env)
        analysis = tune.run(
            "PPO",
            stop={
            "episode_reward_mean": 500
            },
            config={
                "env": "TradingEnv",
                "env_config": {
                    "window_size": 25
                },
                "log_level": "DEBUG",
                "framework": "torch",
                "ignore_worker_failures": True,
                "num_workers": 1,
                "num_gpus": 0,
                "clip_rewards": True,
                "lr": 8e-6,
                "lr_schedule": [
                    [0, 1e-1],
                    [int(1e2), 1e-2],
                    [int(1e3), 1e-3],
                    [int(1e4), 1e-4],
                    [int(1e5), 1e-5],
                    [int(1e6), 1e-6],
                    [int(1e7), 1e-7]
                ],
                "gamma": 0,
                "observation_filter": "MeanStdFilter",
                "lambda": 0.72,
                "vf_loss_coeff": 0.5,
                "entropy_coeff": 0.01
            },
            checkpoint_at_end=True
        )
        # Get checkpoint
        checkpoints = analysis.get_trial_checkpoints_paths(
            trial=analysis.get_best_trial("episode_reward_mean"),
            metric="episode_reward_mean"
        )
        self.checkpoint_path = checkpoints[0][0]
        print('cp={0};'.format(self.checkpoint_path))
        '''

    def evaluate(self):

        # Restore agent
        agent = ppo.PPOTrainer(
            env="TradingEnv",
            config={
                "env_config": {
                    "window_size": 25
                },
                "framework": "torch",
                "log_level": "DEBUG",
                "ignore_worker_failures": True,
                "num_workers": 1,
                "num_gpus": 0,
                "clip_rewards": True,
                "lr": 8e-6,
                "lr_schedule": [
                    [0, 1e-1],
                    [int(1e2), 1e-2],
                    [int(1e3), 1e-3],
                    [int(1e4), 1e-4],
                    [int(1e5), 1e-5],
                    [int(1e6), 1e-6],
                    [int(1e7), 1e-7]
                ],
                "gamma": 0,
                "observation_filter": "MeanStdFilter",
                "lambda": 0.72,
                "vf_loss_coeff": 0.5,
                "entropy_coeff": 0.01
            }
        )
        agent.restore(self.checkpoint_path)
        # Instantiate the environment
        env = self.create_env({
            "window_size": 25
        })

        # Run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()

        while not done:
            action = agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        env.render()

    def create_env(self, config):
        x = np.arange(0, 2*np.pi, 2*np.pi / 1001)
        y = 50*np.sin(3*x) + 100

        x = np.arange(0, 2*np.pi, 2*np.pi / 1000)
        iqc_stream = Stream.source(y, dtype="float").rename("CNY-IQC")

        iqcex = Exchange("iqcex", service=execute_order)(
            iqc_stream
        )
        CNY = Instrument("CNY", 2, "China Yuan")
        IQC = Instrument("IQC", 8, "Iching Quantitative Coin")
        cash = Wallet(iqcex, 100000 * CNY)
        asset = Wallet(iqcex, 0 * IQC)

        portfolio = Portfolio(CNY, [
            cash,
            asset
        ])

        feed = DataFeed([
            iqc_stream,
            iqc_stream.rolling(window=10).mean().rename("fast"),
            iqc_stream.rolling(window=50).mean().rename("medium"),
            iqc_stream.rolling(window=100).mean().rename("slow"),
            iqc_stream.log().diff().fillna(0).rename("lr")
        ])

        reward_scheme = PBR(price=iqc_stream)

        action_scheme = BSH(
            cash=cash,
            asset=asset
        ).attach(reward_scheme)
        # ?????????????????????????????????????????????????????
        self.agent = action_scheme.action_space

        renderer_feed = DataFeed([
            Stream.source(y, dtype="float").rename("price"),
            Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
        ])

        environment = default.create(
            feed=feed,
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            renderer_feed=renderer_feed,
            renderer=PositionChangeChart(),
            window_size=config["window_size"],
            max_allowed_loss=0.6
        )
        return environment