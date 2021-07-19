"""A training script of Soft Actor-Critic on ."""
import argparse
import functools
import logging
import sys
from gym_minigrid.wrappers import *
import gym
import gym.wrappers
import numpy as np
import torch
from pfrl.experiments import StepHook
from torch import distributions, nn
import adversarial_env.adversarial_minigrid

import pfrl
#from diayn.discriminator import Discriminator
#from diayn.vecwrapper import DIAYNWrapper
from args import get_args
from custom_pfrl.train_agent_batch import train_paired
from pfrl import experiments, replay_buffers, utils
from pfrl.nn.lmbda import Lambda


def make_env(args, seed, test, augment_with_z=False):
    env = gym.make(args.env, size=10)
    # Unwrap TimiLimit wrapper
    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env

    # Use different random seeds for train and test envs
    env_seed = 2 ** 32 - 1 - seed if test else seed
    env.seed(int(env_seed))

    # Cast observations to float32 because our model uses float32
    env = pfrl.wrappers.CastObservationToFloat32(env)
    from adversarial_env import wrappers
    env = wrappers.DiscreteToBoxActionSpace(env)
    # Normalize action space to [-1, 1]^n
    env = pfrl.wrappers.NormalizeActionSpace(env)

    if args.monitor:
        out_dir_for_demos = args.outdir
        if augment_with_z is not False:
            out_dir_for_demos = out_dir_for_demos + f"/videos-for-z{augment_with_z}"

        env = pfrl.wrappers.Monitor(
            env, out_dir_for_demos, force=True, video_callable=lambda _: True
        )

    if False:#args.render:
        env = pfrl.wrappers.Render(env, mode="human")

    """
    if augment_with_z is not False:
        from diayn.evalwrapper import DIAYNAugmentationWrapper
        env = DIAYNAugmentationWrapper(env, augment_with_z=augment_with_z, oh_len=1 if not args.diayn_concat_z_oh else args.diayn_n_skills, out_dir=out_dir_for_demos)
    """

    return env


def adv_train(args, ant_agent, ant_env, pro_agent, pro_env):
    args = argparse.Namespace(**vars(args))
    args.env = args.env[0]
    args.outdir = args.outdir+"/adv"

    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def make_batch_env(test, discriminator=None, force_no_diayn=False, is_evaluator=False):
        env = pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, args, process_seeds[idx], test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )

        from adversarial_env.adversarial_minigrid import AdversarialVecWrapper
        env = AdversarialVecWrapper(env)

        """
        if args.diayn_use and force_no_diayn is not True:
            env = DIAYNWrapper(env, discriminator, args.diayn_n_skills, args.diayn_alpha, is_evaluator=is_evaluator, oh_concat=args.diayn_concat_z_oh)
        """
        return env

    """
    discriminator = None
    if args.diayn_use:
        sample_env = make_batch_env(test=False, discriminator=None, force_no_diayn=True)
        obs_space = sample_env.observation_space.shape
        print("Old observation space", sample_env.observation_space)
        print("Old action space", sample_env.action_space)
        del sample_env

        discriminator = Discriminator(
            input_size=obs_space,
            hidden_channels=args.n_hidden_channels,
            hidden_layers=args.n_hidden_layers,
            n_skills=args.diayn_n_skills
        ).cuda()"""

    sample_env = make_batch_env(args)#, discriminator=discriminator)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)
    del sample_env

    action_size = action_space.low.size
    action_space_low = action_space.low
    action_space_high = action_space.high

    #if len(args.load) > 0: fixme
    #    agent.load(args.load)

    from agent import make_agent_policy


    train_paired(
        adv_agent=make_agent_policy(args, obs_space, action_size, action_space_low, action_space_high, train_it=True),
        adv_env=make_batch_env(test=False),
        ant_agent=ant_agent,
        ant_env=ant_env,
        pro_agent=pro_agent,
        pro_env=pro_env,
        outdir=args.outdir,
        steps=args.steps,
        log_interval=args.log_interval,
        max_episode_len=timestep_limit,
    )
