"""A training script of Soft Actor-Critic on ."""
import argparse
import functools
import logging
import sys
import threading

from gym_minigrid.wrappers import *
import gym
import gym.wrappers
import numpy as np
import torch
from pfrl.experiments import StepHook
from torch import distributions, nn

import pfrl
#from diayn.discriminator import Discriminator
#from diayn.vecwrapper import DIAYNWrapper
from adversarial_training import adv_train
from args import get_args
from gonist_training import gon_train
from pfrl import experiments, replay_buffers, utils
from pfrl.nn.lmbda import Lambda


def main():
    args = get_args()

    ant_agent, ant_env = gon_train(args, False)
    pro_agent, pro_env = gon_train(args, True)
    adv_train(args, ant_agent, ant_env, pro_agent, pro_env)

if __name__ == "__main__":
    main()