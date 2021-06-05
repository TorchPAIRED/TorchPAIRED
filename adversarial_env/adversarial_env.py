# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""An environment which is built by a learning adversary.
Has additional functions, step_adversary, and reset_agent. How to use:
1. Call reset() to reset to an empty environment
2. Call step_adversary() to place the goal, agent, and obstacles. Repeat until
   a done is received.
3. Normal RL loop. Use learning agent to generate actions and use them to call
   step() until a done is received.
4. If required, call reset_agent() to reset the environment the way the
   adversary designed it. A new agent can now play it using the step() function.
"""
import random

import gym
import gym_minigrid.minigrid as minigrid
import networkx as nx
from networkx import grid_graph
import numpy as np


class AdversarialWrapper(gym.Wrapper):
  """Grid world where an adversary build the environment the agent plays.
  The adversary places the goal, agent, and up to n_clutter blocks in sequence.
  The action dimension is the number of squares in the grid, and each action
  chooses where the next item should be placed.
  """

  def __init__(self, env, n_clutter=50, random_z_dim=50, max_steps=250):
    super().__init__(env)

    self.random_z_dim = random_z_dim

    self.agent_start_pos = None
    self.goal_pos = None
    self.n_clutter = n_clutter

    # Add two actions for placing the agent and goal.
    self.adversary_max_steps = self.n_clutter + 2

    # Metrics
    self.reset_metrics()

    self.size = env.width # width or height, doesnt matter, it's a square

    # Create spaces for adversary agent's specs.
    self.action_space = (self.size - 2)**2
    self.observation_space = gym.spaces.Discrete(self.adversary_action_dim)

    self.adversary_image_obs_space = gym.spaces.Box(
        low=0,
        high=255,
        shape=(self.size * self.size + self.random_z_dim),  # todo might need to add the timestep like google originally did
        dtype='uint8')

    # NetworkX graph used for computing shortest path
    self.graph = grid_graph(dim=[self.size-2, self.size-2])
    self.wall_locs = []

  def get_obs(self):
    random_z = np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)
    image = self.grid.encode()

    obs = np.append(image[0].flatten(), random_z)
    return obs

  def get_goal_x(self):
    if self.goal_pos is None:
      return -1
    return self.goal_pos[0]

  def get_goal_y(self):
    if self.goal_pos is None:
      return -1
    return self.goal_pos[1]

  def reset_metrics(self):
    self.distance_to_goal = -1
    self.n_clutter_placed = 0
    self.deliberate_agent_placement = -1
    self.passable = -1
    self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1

  def reset(self):
    """Fully resets the environment to an empty grid with no agent or goal."""
    self.graph = grid_graph(dim=[self.size-2, self.size-2])
    self.wall_locs = []

    self.step_count = 0
    self.adversary_step_count = 0

    self.agent_start_dir = self._rand_int(0, 4)

    # Current position and direction of the agent
    self.reset_agent_status()

    self.agent_start_pos = None
    self.goal_pos = None

    # Extra metrics
    self.reset_metrics()

    # Generate the grid. Will be random by default, or same environment if
    # 'fixed_environment' is True.
    self.env.grid = minigrid.Grid(self.size, self.size)
    self.env.grid.wall_rect(0, 0, self.size, self.size)

    return self.get_obs()

  def remove_wall(self, x, y):
    if (x-1, y-1) in self.wall_locs:
      self.wall_locs.remove((x-1, y-1))
    obj = self.grid.get(x, y)
    if obj is not None and obj.type == 'wall':
      self.grid.set(x, y, None)

  def compute_shortest_path(self):
    if self.agent_start_pos is None or self.goal_pos is None:
      return

    self.distance_to_goal = abs(
        self.goal_pos[0] - self.agent_start_pos[0]) + abs(
            self.goal_pos[1] - self.agent_start_pos[1])

    # Check if there is a path between agent start position and goal. Remember
    # to subtract 1 due to outside walls existing in the Grid, but not in the
    # networkx graph.
    self.passable = nx.has_path(
        self.graph,
        source=(self.agent_start_pos[0] - 1, self.agent_start_pos[1] - 1),
        target=(self.goal_pos[0]-1, self.goal_pos[1]-1))
    if self.passable:
      # Compute shortest path
      self.shortest_path_length = nx.shortest_path_length(
          self.graph,
          source=(self.agent_start_pos[0]-1, self.agent_start_pos[1]-1),
          target=(self.goal_pos[0]-1, self.goal_pos[1]-1))
    else:
      # Impassable environments have a shortest path length 1 longer than
      # longest possible path
      self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1

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
    if loc >= self.adversary_action_dim:
      raise ValueError('Position passed to step_adversary is outside the grid.')

    # Add offset of 1 for outside walls
    x = int(loc % (self.width - 2)) + 1
    y = int(loc / (self.width - 2)) + 1
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

      # Goal has already been placed here
      if self.env.grid.get(x, y) is not None:
        # Place agent randomly
        self.agent_start_pos = self.place_one_agent(0, rand_dir=False)
        self.deliberate_agent_placement = 0
      else:
        self.agent_start_pos = np.array([x, y])
        self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)
        self.deliberate_agent_placement = 1

    # Place wall
    elif self.step_count < self.max_steps:
      # If there is already an object there, action does nothing
      if self.env.grid.get(x, y) is None:
        self.put_obj(minigrid.Wall(), x, y)
        self.n_clutter_placed += 1
        self.wall_locs.append((x-1, y-1))

    self.step_count += 1

    # End of episode
    if self.step_count >= self.max_steps:
      done = True
      # Build graph after we are certain agent and goal are placed
      for w in self.wall_locs:
        self.graph.remove_node(w)
      self.compute_shortest_path()

    return self.get_obs(), 0, done, {}