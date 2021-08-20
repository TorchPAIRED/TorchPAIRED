import random

import gym
import gym_minigrid.minigrid as minigrid
import networkx as nx
from networkx import grid_graph
import numpy as np
from pfrl.envs import MultiprocessVectorEnv
from pfrl.wrappers.vector_frame_stack import VectorEnvWrapper


class MinigridAdversarialEnv(minigrid.MiniGridEnv):
  """Grid world where an adversary build the environment the agent plays.
  The adversary places the goal, agent, and up to n_clutter blocks in sequence.
  The action dimension is the number of squares in the grid, and each action
  chooses where the next item should be placed.
  """

  def __init__(self, size, n_clutter=50, random_z_dim=50, max_steps=250):
    self.size = size
    self.random_z_dim = random_z_dim
    self.n_clutter = n_clutter
    self.mission = "generate an env"
    self.passable = True  # gets changed later

    super().__init__(max_steps)

    # Add two actions for placing the agent and goal.
    self.max_steps = self.n_clutter + 2
    self.action_dim = (self.size - 2) ** 2
    self.action_space = gym.spaces.Discrete(self.action_dim)
    self.observation_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(self.size * self.size + self.random_z_dim + 1,),  # last 1 is timestep
        dtype='uint8'
    )

    self.reset()

  def get_obs(self):
    random_z = np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)
    image = self.grid.encode()
    flat_image = image[:,:,0].flatten()

    obs = np.append(flat_image, random_z)
    obs = np.append(obs, self.step_count)

    return obs

  def reset(self):


    """Fully resets the environment to an empty grid with no agent or goal."""
    self.graph = grid_graph(dim=[self.size-2, self.size-2])
    self.step_count = 0

    self.agent_pos = (1,1)  # fake, only there because otherwise printing the env breaks everything
    self.agent_dir = 0
    self.carrying = False

    self.goal_pos = None
    self.goal_pos = (2,2)

    # Generate the grid. Will be random by default, or same environment if
    # 'fixed_environment' is True.
    self.grid = minigrid.Grid(self.size, self.size)
    self.grid.wall_rect(0, 0, self.size, self.size)

    # logging
    self.n_clutter_placed = 0

    return self.get_obs()

  def get_env_configuration(self):
    encoding = self.grid.encode()

    from adversarial_env.configuration import EnvConfiguration
    conf = EnvConfiguration(encoding, self.agent_pos, self.agent_dir, self.goal_pos, self.carrying, self.passable, self.size)
    return conf

  def remove_wall(self, x, y):
    obj = self.grid.get(x, y)
    if obj is not None and obj.type == 'wall':
      self.grid.set(x, y, None)

  def compute_shortest_path(self):
    if self.agent_pos is None or self.goal_pos is None:
      return

    agent_pos = self.agent_pos[0]-1, self.agent_pos[1]-1
    goal_pos = self.goal_pos[0]-1, self.goal_pos[1]-1

    """
    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 6))
    pos = {(x, y): (y, -x) for x, y in self.graph.nodes()}
    nx.draw(self.graph, pos=pos,
            node_color='lightgreen',
            with_labels=True,
            node_size=600)
    plt.show()
    """

    self.distance_to_goal = abs(goal_pos[0] - agent_pos[0]) \
                            + abs(goal_pos[1] - agent_pos[1])

    # Check if there is a path between agent start position and goal. Remember
    # to subtract 1 due to outside walls existing in the Grid, but not in the
    # networkx graph.
    self.passable = nx.has_path(
        self.graph,
        source=(agent_pos[0], agent_pos[1]),
        target=(goal_pos[0], goal_pos[1]))
    if self.passable:
      # Compute shortest path
      self.shortest_path_length = nx.shortest_path_length(
          self.graph,
          source=(agent_pos[0], agent_pos[1]),
          target=(goal_pos[0], goal_pos[1]))
    else:
      # Impassable environments have a shortest path length 1 longer than
      # longest possible path
      self.shortest_path_length = (self.size - 2) * (self.size - 2) + 1

  def place_agent_at_pos(self, pos, rand_dir=True):
    x, y = pos

    was_at_pos = self.grid.get(x, y)
    self.deliberate_agent_placement = 1
    if was_at_pos is not None:
      x, y = self.place_obj(None)   # returns a free pos
      self.deliberate_agent_placement = 0

    self.agent_pos = x, y
    if rand_dir:
      self.agent_dir = self._rand_int(0, 4)

    self.grid.set(x, y, None)


  def step(self, loc):
    """The adversary gets n_clutter + 2 moves to place the goal, agent, blocks.
    The action space is the number of possible squares in the grid. The squares
    are numbered from left to right, top to bottom.
    Args:
      loc: An integer specifying the location to place the next object which
        must be decoded into x, y coordinates.
    Returns:
      Standard RL observation, reward (always 0), done, and info
    """
    if loc >= self.action_dim:
      raise ValueError('Position passed to step_adversary is outside the grid.')

    # Add offset of 1 for outside walls
    x = int(loc % (self.size - 2)) + 1
    y = int(loc / (self.size - 2)) + 1
    done = False

    should_choose_goal = self.step_count == 0
    should_choose_agent = self.step_count == 1

    # Place goal
    if should_choose_goal:
      self.remove_wall(x, y)  # Remove any walls that might be in this loc
      self.put_obj(minigrid.Goal(), x, y)
      self.goal_pos = (x, y)

    # Place the agent
    elif should_choose_agent:
      self.remove_wall(x, y)  # Remove any walls that might be in this loc
      self.place_agent_at_pos((x,y),False)

    # Place wall
    elif self.step_count < self.max_steps:
      # If there is already an object there, action does nothing
      if self.grid.get(x, y) is None and self.agent_pos != (x,y):
        self.put_obj(minigrid.Wall(), x, y)
       # print(f"placed wall at {x,y}")

        # logging
        self.n_clutter_placed += 1
        self.graph.remove_node((x-1,y-1)) #fixme this is probably wrong

    self.step_count += 1

    # End of episode
    info = {}
    if self.step_count >= self.max_steps:
      done = True
      self.compute_shortest_path()

      info["configuration"] = self.get_env_configuration()


    return self.get_obs(), 0, done, info

class AdversarialVecWrapper(VectorEnvWrapper):
  def __init__(self, env):
    super().__init__(env)
    self.reset()

  def step(self, action):
    obss, rews, dones, infos = self.env.step(action)

    return obss, rews, dones, infos

  def reset(self, mask=None):
    return self.env.reset(mask)

gym.envs.register(
     id='AdversarialMinigrid-v0',
     entry_point='adversarial_env.adversarial_minigrid:AdversarialEnv',
     max_episode_steps=1000,
)