#
import gym
#
from apps.drl.c06.envs.max_and_skip_env import MaxAndSkipEnv
from apps.drl.c06.envs.fire_reset_env import FireResetEnv
from apps.drl.c06.envs.process_frame_84_obs import ProcessFrame84Obs
from apps.drl.c06.envs.image_to_pytorch_obs import ImageToPytorchObs
from apps.drl.c06.envs.buffer_wrapper_obs import BufferWrapperObs
from apps.drl.c06.envs.scaled_folat_frame_obs import ScaledFloatFrameObs

class EnvManager(object):
    @staticmethod
    def make_env(env_name):
        env = gym.make(env_name)
        env = MaxAndSkipEnv(env)
        env = FireResetEnv(env)
        env = ProcessFrame84Obs(env)
        env = ImageToPytorchObs(env)
        env = BufferWrapperObs(env, 4)
        return ScaledFloatFrameObs(env)