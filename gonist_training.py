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
from pfrl.experiments.evaluator import save_agent
from torch import distributions, nn
import adversarial_env.adversarial_minigrid

import pfrl
#from diayn.discriminator import Discriminator
#from diayn.vecwrapper import DIAYNWrapper
from args import get_args
from custom_envs.get_all_custom_envs import get_all_custom_envs
from pfrl import experiments, replay_buffers, utils
from pfrl.nn.lmbda import Lambda

#from utils import make_n_hidden_layers


def make_env(args, seed, test, force_monitor=False, pre_instantiated=None):
    if pre_instantiated is None:
        env = gym.make(args.env, size=10)
        # Unwrap TimiLimit wrapper
    else:
        env = pre_instantiated

    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env

    # Use different random seeds for train and test envs
    env_seed = 2 ** 32 - 1 - seed if test else seed
    env.seed(int(env_seed))

    from adversarial_env import wrappers
    env = wrappers.WalkingMinigrid(env)
    env = wrappers.FlatsliceWrapper(env)

    # Cast observations to float32 because our model uses float32
    env = pfrl.wrappers.CastObservationToFloat32(env)

    env = wrappers.DiscreteToBoxActionSpace(env)


    # Normalize action space to [-1, 1]^
    from adversarial_env.wrappers import NormalizeActionSpace
    env = NormalizeActionSpace(env)

    if force_monitor is not False:
        out_dir_for_demos = args.outdir
        out_dir_for_demos = out_dir_for_demos + f"/videos-for-{force_monitor}"

        env = pfrl.wrappers.Monitor(
            env, out_dir_for_demos, force=True, video_callable=lambda _: True
        )

    if args.render:
        env = wrappers.MinigridRender(env, mode="human")

    """
    if augment_with_z is not False:
        from diayn.evalwrapper import DIAYNAugmentationWrapper
        env = DIAYNAugmentationWrapper(env, augment_with_z=augment_with_z, oh_len=1 if not args.diayn_concat_z_oh else args.diayn_n_skills, out_dir=out_dir_for_demos)
    """

    return env


def gon_train(args, is_protag):
    args = argparse.Namespace(**vars(args))
    args.env = args.env[1]
    args.outdir = args.outdir + ("/pro" if is_protag else "/ant")

    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def make_batch_env(test, initial_configuration, force_monitor=False):#discriminator=None, force_no_diayn=False, is_evaluator=False):
        env = pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, args, process_seeds[idx], test, force_monitor)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )

        from adversarial_env.adversarial_minigrid import AdversarialVecWrapper
        from adversarial_env.settable_minigrid import SettableVecWrapper
        env = SettableVecWrapper(env, initial_configuration, for_protag=is_protag)

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
    from adversarial_env.adversarial_minigrid import AdversarialEnv
    initial_confuguration = AdversarialEnv(size=10).get_env_configuration()
    #make_env(args, 0, False, augment_with_z=False).step(conf)

    sample_env = make_batch_env(args, [initial_confuguration]*args.num_envs)#, discriminator=discriminator)
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    action_size = action_space.low.size
    action_space_low = action_space.low
    action_space_high = action_space.high

    #if len(args.load) > 0: fixme
    #    agent.load(args.load)

    # jank!
    import threading

    from agent import make_agent_policy
    agent = make_agent_policy(args, obs_space, action_size, action_space_low, action_space_high)
    env = make_batch_env(test=False, initial_configuration=[initial_confuguration]*args.num_envs)
    display_env = make_batch_env(test=True, initial_configuration=[initial_confuguration]*args.num_envs, force_monitor="display_adv")
    eval_envs = []
    for i, pre_instantiated_env in enumerate(get_all_custom_envs("./custom_envs")):
        eval_envs.append(make_env(args, process_seeds[0]+i, True, force_monitor="evals", pre_instantiated=pre_instantiated_env))

    # todo make custom agent for lstm
    # see

    timestep_limit = make_env(args, 0, False, force_monitor=False).unwrapped.max_steps

    return AgentTrainer(agent, env, display_env, eval_envs, max_episode_len=timestep_limit, checkpoint_freq=args.eval_interval, outdir=args.outdir, logger=logging.getLogger(__name__), is_pro=is_protag, log_interval=10000)

class AgentTrainer:
    def __init__(self, agent, env, display_env, eval_envs, max_episode_len, checkpoint_freq, outdir, logger, is_pro=True, log_interval=10000):
        self.agent = agent

        self.env = env
        self.display_env = display_env
        self.eval_envs = eval_envs
        self.best_eval_scores = {f"{eval_env.env_name}": -np.inf for eval_env in self.eval_envs}

        self.max_episode_len = max_episode_len
        self.checkpoint_freq = checkpoint_freq
        self.outdir = outdir
        self.logger = logger
        self.is_pro = is_pro
        self.log_interval = log_interval

        num_envs = env.num_envs
        self.num_envs = env.num_envs

        # episode_r = np.zeros(num_envs, dtype=np.float64)
        self.episode_len = np.zeros(num_envs, dtype="i")

        self.last_obss = env.reset()

        self.t = 0
        if hasattr(agent, "t"):
            agent.t = 0

        self.waiters_rews = np.zeros(num_envs)
        self.waiters_waiting = np.zeros(num_envs)
        self.waiters_obss = np.zeros((num_envs, env.observation_space.low.shape[0]))
        self.episode_idx = np.zeros(num_envs)

        self.total_episode_idx = np.zeros(num_envs)

        eval_stats_history = []  # List of evaluation episode stats dict

    def init_episodes(self, configuration):
        self.current_configuration = configuration
        self.env.set(configuration)

        self.episode_idx = np.zeros(self.num_envs, dtype="i")
        self.rew_counter = np.zeros(self.num_envs)

    def get_step(self):
        actions = self.agent.batch_act(self.last_obss, skip_mask=self.waiters_waiting)
        obss, rs, dones, infos = self.env.step(actions)
        rs = np.array(rs)
        dones = np.array(dones)
        obss = np.array(obss)

        # if True in (rs != 0):
        #     print(rs)

        self.rew_counter += rs
        self.last_obss = obss
        self.last_dones = dones
        self.last_rs = rs
        self.last_infos = infos

        # self.episode_r += rs9
        self.episode_len += 1

        update_waiters_mask = np.logical_and(self.waiters_waiting == False, dones)
        self.waiters_rews[update_waiters_mask] = rs[update_waiters_mask]
        self.waiters_obss[update_waiters_mask] = obss[update_waiters_mask]
        self.waiters_waiting[update_waiters_mask] = True

        return self.waiters_rews, self.waiters_waiting

    def do_update(self, unlocked_waiters, regret):
        if self.max_episode_len is None:
            resets = np.zeros(self.num_envs, dtype=bool)
        else:
            resets = self.episode_len == self.max_episode_len
        resets = np.logical_or(
            resets, [info.get("needs_reset", False) for info in self.last_infos]
        )
        resets[unlocked_waiters] = True

        obss = self.last_obss
        obss[unlocked_waiters] = self.waiters_obss[unlocked_waiters]

        rews = regret

        dones = self.last_dones
        dones[unlocked_waiters] = True
        #if True in (self.waiters_rews != 0):
        #    i=0
        #    i = 0

        self.waiters_waiting[unlocked_waiters] = False
        self.waiters_obss[unlocked_waiters] = 0
        self.waiters_rews[unlocked_waiters] = 0

        if np.sum(self.waiters_waiting) != self.num_envs:
            from utils import timer
            #with timer(f"obs {self.is_pro}", also=f"unlocked waiters: {unlocked_waiters}"):
                # Agent observes the consequences
            self.agent.batch_observe(obss, rews, dones, resets, skip_mask=self.waiters_waiting) # skip waiters

        # dones[self.waiters_rews != 0] = False
        # resets[self.waiters_rews != 0] = False

        #with timer("mid"):
        end = np.logical_or(resets, dones)
        not_end = np.logical_not(end)

        self.episode_idx += end
        self.total_episode_idx += end

        for i in range(self.num_envs):
            if self.waiters_waiting[i]:
                continue

            self.t += 1
            if self.checkpoint_freq and self.t % self.checkpoint_freq == 0:
                self.display_env.set(self.current_configuration)
                #self.display_env.step(np.ones((1,)))

                save_agent(self.agent, self.t, self.outdir, self.logger)

                from pfrl.experiments.evaluator import eval_performance
                for env in self.eval_envs:
                    env_name = env.unwrapped.env_name
                   # env.reset()

                    """
                    from PIL import Image
                    img = Image.fromarray(env.render(mode="inhuman, lol"))
                    img = img.resize((300, 300))
                    img.show(title="conf")

                    env.step(-np.ones(1,))
                    #env.step(np.ones((1,)))

                    img = Image.fromarray(env.render(mode="inhuman, lol"))
                    img = img.resize((300, 300))
                    img.show(title="conf")
"""

                    eval_returns = eval_performance(env, self.agent, None,5, self.max_episode_len)
                    if self.best_eval_scores[env_name] < eval_returns["mean"]:
                        print(f"best score for {env_name} updated: {self.best_eval_scores[env_name]} ---> {eval_returns['mean']}")
                        self.best_eval_scores[env_name] = eval_returns["mean"]
                    else:
                        print(f"best score for {env_name}: {self.best_eval_scores[env_name]}")


        if (
                self.log_interval is not None
                and self.t >= self.log_interval
                and self.t % self.log_interval < self.num_envs
        ):
            self.logger.info(
                "outdir:{} step:{} episode:{} last_R: {} average_R:{}".format(  # NOQA
                    self.outdir,
                    self.t,
                    np.sum(self.episode_idx),
                    self.rew_counter[-1],
                    np.mean(self.rew_counter),
                )
            )
            self.logger.info("statistics: {}".format(self.agent.get_statistics()))

        self.last_obss = self.env.reset(not_end)
        return self.episode_idx