#
import numpy as np
import gym

class ImageToPytorchObs(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPytorchObs, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)