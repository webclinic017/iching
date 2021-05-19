#
import numpy as np
import gym

class ScaledFloatFrameObs(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0