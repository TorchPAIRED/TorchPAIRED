import gym
import numpy as np

class FlatsliceWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.env.agent_view_size*self.env.agent_view_size,),
            dtype='uint8'
        )

    def observation(self, observation):
        return np.append(observation["image"][0].flatten(), observation["direction"])