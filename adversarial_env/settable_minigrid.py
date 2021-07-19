import gym
import gym_minigrid.minigrid
import numpy as np
from gym_minigrid import minigrid
from pfrl.envs import MultiprocessVectorEnv


class SettableMinigrid(gym_minigrid.minigrid.MiniGridEnv):
    def __init__(self, size, max_steps=250):
        self.set_grid = minigrid.Grid(size, size)
        self.set_agent_pos = (0, 0)
        self.set_agent_dir = 0
        self.set_goal_pos = (1, 1)
        self.set_carrying = False
        self.mission = 'reach the goal'
        self.passable = True

        self.paired_done = False

        super().__init__(size, max_steps=max_steps)


    def set(self, configuration):
        new_grid, agent_pos, agent_dir, goal_pos, carrying = configuration.get_configuration()

        from gym_minigrid.minigrid import Grid

        self.set_grid = Grid.decode(new_grid)[0] # already implements deepcopy
        self.set_agent_pos = agent_pos
        self.set_agent_dir = agent_dir
        self.set_goal_pos = goal_pos
        self.set_carrying = carrying
        self.passable = configuration.get_passable()

        return self.reset()

    def step(self, action):
        try:
            obs = self.set(action)
            return obs, 0, False, {}
        except Exception as e:
            #print(e)
            pass

        ##print(action)
        action = int(action[0])
        #print(action)

        obs, rew, done, info = super().step(action)
        if done:
            self.paired_done = True

        return obs, rew, done, info

    def _gen_grid(self, _w, _h):
        env = self

        env.grid = self.set_grid
        env.agent_pos = self.set_agent_pos
        env.agent_dir = self.set_agent_dir
        env.goal_pos = self.set_goal_pos
        env.carrying = self.set_goal_pos

from pfrl.wrappers.vector_frame_stack import VectorEnvWrapper
class SettableVecWrapper(VectorEnvWrapper):
    def __init__(self, env, initial_configuration, for_protag):
        super().__init__(env, )
        self.num_envs = env.num_envs

        self.reset_counter = np.zeros((self.num_envs,))
        self.reset()

        self.rewards = np.zeros((self.num_envs,))
        self.for_protag = for_protag
        self.n_steps_without_set = 0

        self.set(initial_configuration)

        self.reset()

    def step(self, action):
        #for i in range(len(action)):
        #    action[i] = int(action[i])

        obss, rews, dones, infos = self.env.step(action)

        for i,rew in enumerate(rews):
            self.rewards[i] += rew

        self.n_steps_without_set += 1
        #print(self.n_steps_without_set)


        return obss, rews, dones, infos

    def reset(self, mask=None):
        """
        if np.count_nonzero(self.reset_counter) == self.num_envs:
            self.reset_counter = np.zeros((self.num_envs,))
            configurations = self.pipe.get_configuration(from_protag=self.for_protag, reward=self.rewards)
            self.set(configurations)
            self.rewards = np.zeros((self.env.num_envs,))


        elif mask is not None:
            self.reset_counter[np.logical_not(mask)] += 1"""

        return self.env.reset(mask)

    def set(self, configurations):
        self.step(configurations)
        self.reset()



gym.envs.register(
     id='SettableMinigrid-v0',
     entry_point='adversarial_env.settable_minigrid:SettableMinigrid',
     max_episode_steps=1000,
)