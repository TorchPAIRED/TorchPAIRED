import argparse
import functools
import logging
import sys

import gym
import gym.wrappers
import numpy as np
import torch
from torch import distributions, nn

import pfrl
#from diayn.discriminator import Discriminator
#from diayn.vecwrapper import DIAYNWrapper
from pfrl import experiments, replay_buffers, utils
from pfrl.nn.lmbda import Lambda

from utils import get_squashed_diagonal_gaussian_head_fun


def make_agent_policy(args, obs_space, action_size, action_space, train_it=True):
    squashed_diagonal_gaussian_head = get_squashed_diagonal_gaussian_head_fun(action_size)

    from utils import make_n_hidden_layers
    policy = nn.Sequential(  # todo make this recurrent
        nn.Linear(obs_space.low.size, args.n_hidden_channels),
        nn.ReLU(),
        *make_n_hidden_layers(args.n_hidden_layers, args.n_hidden_channels),
        nn.Linear(args.n_hidden_channels, action_size * 2),
        Lambda(squashed_diagonal_gaussian_head),
    )
    torch.nn.init.xavier_uniform_(policy[0].weight)
    torch.nn.init.xavier_uniform_(policy[2].weight)
    torch.nn.init.xavier_uniform_(policy[4].weight)
    policy_optimizer = torch.optim.Adam(
        policy.parameters(), lr=args.lr, eps=args.adam_eps
    )

    def make_q_func_with_optimizer():
        q_func = nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_space.low.size + action_size, args.n_hidden_channels),
            nn.ReLU(),
            *make_n_hidden_layers(args.n_hidden_layers, args.n_hidden_channels),
            nn.Linear(args.n_hidden_channels, 1),
        )
        torch.nn.init.xavier_uniform_(q_func[1].weight)
        torch.nn.init.xavier_uniform_(q_func[3].weight)
        torch.nn.init.xavier_uniform_(q_func[5].weight)
        q_func_optimizer = torch.optim.Adam(
            q_func.parameters(), lr=args.lr, eps=args.adam_eps
        )
        return q_func, q_func_optimizer

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()

    rbuf = replay_buffers.ReplayBuffer(10 ** 6, num_steps=args.n_step_return)

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space_low, action_space_high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    agent = pfrl.agents.SoftActorCritic(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=args.discount,
        update_interval=args.update_interval,
        replay_start_size=args.replay_start_size,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        burnin_action_func=burnin_action_func,
        entropy_target=-action_size,
        temperature_optimizer_lr=args.lr,
    )

    return agent