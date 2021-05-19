#
import time
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#
from apps.drl.c06.app_config import AppConfig
from apps.drl.c06.experience_buffer import ExperienceBuffer
from apps.drl.c06.pong_agent import PongAgent
from apps.drl.c06.envs.env_manager import EnvManager
from apps.drl.c06.pong_dqn_model import PongDqnModel

class C06App(object):
    def __init__(self):
        self.name = 'apps.drl.c06.c06_app.C06App'

    def startup(self, args={}):
        self.train()

    def train(self):
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument("--cuda", default=False,
                            action="store_true", help="Enable cuda")
        parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                            help="Name of the environment, default=" +
                                DEFAULT_ENV_NAME)
        args = parser.parse_args()
        '''
        device = torch.device("cuda:0") # if args.cuda else "cpu")
        env = EnvManager.make_env(AppConfig.DEFAULT_ENV_NAME)
        net = PongDqnModel(env.observation_space.shape,
                            env.action_space.n).to(device)
        tgt_net = PongDqnModel(env.observation_space.shape,
                                env.action_space.n).to(device)
        print(net)
        buffer = ExperienceBuffer(AppConfig.REPLAY_SIZE)
        agent = PongAgent(env, buffer)
        epsilon = AppConfig.EPSILON_START

        optimizer = optim.Adam(net.parameters(), lr=AppConfig.LEARNING_RATE)
        total_rewards = []
        frame_idx = 0
        ts_frame = 0
        ts = time.time()
        best_m_reward = None

        while True:
            frame_idx += 1
            epsilon = max(AppConfig.EPSILON_FINAL, AppConfig.EPSILON_START -
                        frame_idx / AppConfig.EPSILON_DECAY_LAST_FRAME)

            reward = agent.play_step(net, epsilon, device=device)
            if reward is not None:
                total_rewards.append(reward)
                speed = (frame_idx - ts_frame) / (time.time() - ts)
                ts_frame = frame_idx
                ts = time.time()
                m_reward = np.mean(total_rewards[-100:])
                print("%d: done %d games, reward %.3f, "
                    "eps %.2f, speed %.2f f/s" % (
                    frame_idx, len(total_rewards), m_reward, epsilon,
                    speed
                ))
                '''
                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("reward_100", m_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)
                '''
                if best_m_reward is None or best_m_reward < m_reward:
                    torch.save(net.state_dict(), AppConfig.DEFAULT_ENV_NAME +
                            "-best_%.0f.dat" % m_reward)
                    if best_m_reward is not None:
                        print("Best reward updated %.3f -> %.3f" % (
                            best_m_reward, m_reward))
                    best_m_reward = m_reward
                if m_reward > AppConfig.MEAN_REWARD_BOUND:
                    print("Solved in %d frames!" % frame_idx)
                    break

            if len(buffer) < AppConfig.REPLAY_START_SIZE:
                continue

            if frame_idx % AppConfig.SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())

            optimizer.zero_grad()
            batch = buffer.sample(AppConfig.BATCH_SIZE)
            loss_t = C06App.calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()

    def run(self):
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--model", required=True,
                            help="Model file to load")
        parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                            help="Environment name to use, default=" +
                                DEFAULT_ENV_NAME)
        parser.add_argument("-r", "--record", help="Directory for video")
        parser.add_argument("--no-vis", default=True, dest='vis',
                            help="Disable visualization",
                            action='store_false')
        args = parser.parse_args()
        '''
        vis_mode = True
        env = EnvManager.make_env(AppConfig.DEFAULT_ENV_NAME)
        record_mode = False
        #if record_mode:
        #    env = gym.wrappers.Monitor(env, args.record)
        net = PongDqnModel(env.observation_space.shape,
                            env.action_space.n)
        model_file = ''
        state = torch.load(model_file, map_location=lambda stg, _: stg)
        net.load_state_dict(state)

        state = env.reset()
        total_reward = 0.0
        c = collections.Counter()

        while True:
            start_ts = time.time()
            if vis_mode:
                env.render()
            state_v = torch.tensor(np.array([state], copy=False))
            q_vals = net(state_v).data.numpy()[0]
            action = np.argmax(q_vals)
            c[action] += 1
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
            if vis_mode:
                delta = 1/AppConfig.FPS - (time.time() - start_ts)
                if delta > 0:
                    time.sleep(delta)
        print("Total reward: %.2f" % total_reward)
        print("Action counts:", c)
        if record_mode:
            env.env.close()

    @staticmethod
    def calc_loss(batch, net, tgt_net, device="cpu"):
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(np.array(
            states, copy=False)).to(device)
        next_states_v = torch.tensor(np.array(
            next_states, copy=False)).to(device)
        actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        state_action_values = net(states_v).gather(
            1, actions_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = tgt_net(next_states_v).max(1)[0]
            next_state_values[done_mask] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * AppConfig.GAMMA + \
                                    rewards_v
        return nn.MSELoss()(state_action_values,
                            expected_state_action_values)