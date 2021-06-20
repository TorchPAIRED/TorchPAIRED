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

#from utils import make_n_hidden_layers
from pipe import AdvPipe


def main():
    args = get_args()
    pipe = AdvPipe()

    protag_thread = threading.Thread(target=gon_train, args=[args, pipe, True])
    antag_thread = threading.Thread(target=gon_train, args=[args, pipe, False])

    protag_thread.start()
    antag_thread.start()

    adv_thread = threading.Thread(target=adv_train, args=[args, pipe])
    adv_thread.start()


if __name__ == "__main__":
    main()