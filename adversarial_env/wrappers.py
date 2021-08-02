import threading

import gym
import numpy as np

class FlatsliceWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.env.agent_view_size*self.env.agent_view_size+1,),   # direction
            dtype='uint8'
        )

    def observation(self, observation):
        return np.append(observation["image"][:,:,0].flatten(), observation["direction"])

class DiscreteToBoxActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.did_conversion = False
        if isinstance(env.action_space, gym.spaces.Discrete):
            size = env.action_space.n
            self.original_size = size
            self.did_conversion = True
            self.action_space = gym.spaces.Box(0, size, shape=(1,))
        assert isinstance(self.action_space, gym.spaces.Box)

    def action(self, action):
        from adversarial_env.configuration import EnvConfiguration
        if isinstance(action, EnvConfiguration):
            return action  # jank for settable env...

        if not self.did_conversion:
            return action

        if action == self.original_size:
            return action-1

        from math import floor
        action = floor(action)
        return action

class WalkingMinigrid(gym.ActionWrapper):
    # prevent thinking time being spent on NOOPs

    def __init__(self, env):
        super().__init__(env)
        from gym_minigrid.minigrid import MiniGridEnv
        assert isinstance(env, MiniGridEnv)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, action):
        return action


class NormalizeActionSpace(gym.ActionWrapper):
    #"Normalize a Box action space to [-1, 1]^n."

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        self.action_space = gym.spaces.Box(
            low=-np.ones_like(env.action_space.low),
            high=np.ones_like(env.action_space.low),
        )

    def action(self, action):
        from adversarial_env.configuration import EnvConfiguration
        if isinstance(action, EnvConfiguration):
            return action #jank for settable env...

        # action is in [-1, 1]
        action = action.copy()

        # -> [0, 2]
        action += 1

        # -> [0, orig_high - orig_low]
        action *= (self.env.action_space.high - self.env.action_space.low) / 2

        # -> [orig_low, orig_high]
        return action + self.env.action_space.low


import gym

import pyglet
class MinigridRender(gym.Wrapper):
    """Render env by calling its render method.

    Args:
        env (gym.Env): Env to wrap.
        **kwargs: Keyword arguments passed to the render method.
    """

    def __init__(self, env, **kwargs):
        super().__init__(env)
        self._kwargs = kwargs
        self.window = pyglet.window.Window()

    def do_render(self):
        img = self.env.render(**self._kwargs)
        import cv2
        cv2.imshow("img", img)
        cv2.waitKey(2)

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        self.do_render()
        return ret

    def step(self, action):
        ret = self.env.step(action)
        self.do_render()
        return ret
