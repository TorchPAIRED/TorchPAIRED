import gym
import gym_minigrid.minigrid
from gym_minigrid import minigrid


class SettableMinigrid(gym_minigrid.minigrid.MiniGridEnv):
    def __init__(self, size, max_steps):
        self.set_grid = minigrid.Grid(size, size)
        self.set_agent_pos = (0, 0)
        self.set_agent_dir = 0
        self.set_goal_pos = (1, 1)
        self.set_carrying = False
        self.mission = 'reach the goal'
        super().__init__(size, max_steps=max_steps)


    def set(self, new_grid, agent_pos, agent_dir, goal_pos, carrying):
        self.set_grid = new_grid.copy() # already implements deepcopy
        self.set_agent_pos = agent_pos
        self.set_agent_dir = agent_dir
        self.set_goal_pos = goal_pos
        self.set_carrying = carrying

        self.reset()

    def _gen_grid(self, _w, _h):
        env = self

        env.grid = self.set_grid
        env.agent_pos = self.set_agent_pos
        env.agent_dir = self.set_agent_dir
        env.goal_pos = self.set_goal_pos
        env.carrying = self.set_goal_pos

gym.envs.register(
     id='SettableMinigrid-v0',
     entry_point='adversarial_env:SettableMinigrid',
     max_episode_steps=1000,
)