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

import pfrl
#from diayn.discriminator import Discriminator
#from diayn.vecwrapper import DIAYNWrapper
from pfrl import experiments, replay_buffers, utils
from pfrl.nn.lmbda import Lambda

#from utils import make_n_hidden_layers


def make_env(args, seed, test, augment_with_z=False):
    env = gym.make(args.env)
    # Unwrap TimiLimit wrapper
    if isinstance(env, gym.wrappers.TimeLimit):
        env = env.env

    # Use different random seeds for train and test envs
    env_seed = 2 ** 32 - 1 - seed if test else seed
    env.seed(int(env_seed))

    from adversarial_env.wrappers import FlatsliceWrapper
    env = FlatsliceWrapper(env)

    # Cast observations to float32 because our model uses float32
    env = pfrl.wrappers.CastObservationToFloat32(env)
    # Normalize action space to [-1, 1]^n
    # fixme env = pfrl.wrappers.NormalizeActionSpace(env)
    if args.monitor:
        out_dir_for_demos = args.outdir
        if augment_with_z is not False:
            out_dir_for_demos = out_dir_for_demos + f"/videos-for-z{augment_with_z}"

        env = pfrl.wrappers.Monitor(
            env, out_dir_for_demos, force=True, video_callable=lambda _: True
        )
    if args.render:
        env = pfrl.wrappers.Render(env, mode="human")

    if augment_with_z is not False:
        from diayn.evalwrapper import DIAYNAugmentationWrapper
        env = DIAYNAugmentationWrapper(env, augment_with_z=augment_with_z, oh_len=1 if not args.diayn_concat_z_oh else args.diayn_n_skills, out_dir=out_dir_for_demos)

    return env


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default="MiniGrid-Empty-8x8-v0",
        help="OpenAI Gym env to perform algorithm on.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of envs run in parallel."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10 ** 7,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=20,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=25000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=10000,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=1,
        help="Interval in timesteps between model updates.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with Monitor to write videos."
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="Interval in timesteps between outputting log messages during training",
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    parser.add_argument(
        "--n-hidden-channels",
        type=int,
        default=300,    # https://github.com/pfnet/pfrl/blob/44bf2e483f5a2f30be7fd062545de306247699a1/examples/gym/train_reinforce_gym.py#L84
        help="Number of hidden channels of NN models.",
    )
    parser.add_argument(
        "--n-hidden-layers",
        type=int,
        default=1,
        # https://github.com/pfnet/pfrl/blob/44bf2e483f5a2f30be7fd062545de306247699a1/examples/gym/train_reinforce_gym.py#L84
        help="Number of hidden channels of NN models.",
    )
    parser.add_argument("--discount", type=float, default=0.98, help="Discount factor.")
    parser.add_argument("--n-step-return", type=int, default=3, help="N-step return.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--adam-eps", type=float, default=1e-1, help="Adam eps.")

    parser.add_argument(
        "--diayn-use",
        action="store_true",
        default=False,
        help="Wether or not we should use diayn",
    )
    parser.add_argument(
        "--diayn-n-skills",
        type=int,
        default=50,
        help="Number of skills to train",
    )
    parser.add_argument(
        "--diayn-concat-z-oh",
        action="store_true",
        default=False,
        help="If true, will use a one-hot of z to augment the observation instead of just z's value.",
    )
    parser.add_argument(
        "--diayn-alpha",
        type=float,
        default=10.0,
        help="How much to scale the intrinsic reward by",
        # note: normally, this is done to the max entropy objective. In our version, we use PFRL, so there's a bit of
        # infrastructure obfuscating the objective. Instead, we scale the other term (the expectation, aka the intrisic
        # reward). The alpha=0.1 suggested in the DIAYN paper thus becomes 10.0, i.e. 1/0.1.
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    print("Output files are saved in {}".format(args.outdir))

    with open(args.outdir + "/args.txt", "w") as f:
        f.write("\n".join(str(args).split(",")))

    # Set a random seed used in PFRL
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

        if args.diayn_use and force_no_diayn is not True:
            env = DIAYNWrapper(env, discriminator, args.diayn_n_skills, args.diayn_alpha, is_evaluator=is_evaluator, oh_concat=args.diayn_concat_z_oh)
        return env

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
        ).cuda()

    sample_env = make_batch_env(args, discriminator=discriminator)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)
    del sample_env

    action_size = action_space.n






    #if len(args.load) > 0: fixme
    #    agent.load(args.load)



    if args.demo:
        if args.diayn_use:
            for z in range(args.diayn_n_skills):
                eval_env = make_env(args, seed=0, test=True, augment_with_z=z)
                eval_stats = experiments.eval_performance(
                    env=eval_env,
                    agent=agent,
                    n_steps=None,
                    n_episodes=args.eval_n_runs,
                    max_episode_len=timestep_limit,
                )
                print(
                    "z: {} n_runs: {} mean: {} median: {} stdev {}".format(
                        z,
                        args.eval_n_runs,
                        eval_stats["mean"],
                        eval_stats["median"],
                        eval_stats["stdev"],
                    )
                )
        else:

            eval_env = make_env(args, seed=0, test=True)
            eval_stats = experiments.eval_performance(
                env=eval_env,
                agent=agent,
                n_steps=None,
                n_episodes=args.eval_n_runs,
                max_episode_len=timestep_limit,
            )
            print(
                "n_runs: {} mean: {} median: {} stdev {}".format(
                    args.eval_n_runs,
                    eval_stats["mean"],
                    eval_stats["median"],
                    eval_stats["stdev"],
                )
            )

    else:
        # jank!
        import threading

        def agent_thread():
            agent_lock = threading.Lock()
            agent_lock.acquire()  # starts locked

            designer_lock = threading.Lock()
            designer_lock.acquire()  # starts locked

            class StopAgentsHook(StepHook):
                def __call__(self, env, agent, step):
                    if env.done: # this agent is done! We need to stop it.
                        designer_lock.release() # tell designer it's free to go
                        agent_lock.acquire()    # go to sleep

            train_envs = make_batch_env(test=False, discriminator=discriminator)
            eval_envs = make_batch_env(test=True, discriminator=discriminator, is_evaluator=True)

            def thread_func():
                from agent import make_agent_policy

                agent_lock.acquire()    # since it starts locked, this is a dead stop until the designer is done

                experiments.train_agent_batch_with_evaluation(
                    agent=make_agent_policy(args, obs_space, action_size, action_space, train_it=True),
                    env=train_envs,
                    eval_env=eval_envs,
                    outdir=args.outdir,
                    steps=args.steps,
                    eval_n_steps=None,
                    eval_n_episodes=args.eval_n_runs,
                    eval_interval=args.eval_interval,
                    log_interval=args.log_interval,
                    max_episode_len=timestep_limit,
                    use_tensorboard=True,
                    step_hooks=[StopAgentsHook()]
                )
            thread = threading.Thread(target=thread_func, args=[], daemon=True)
            thread.start()

            return agent_lock, train_envs, eval_envs, designer_lock
        prot_lock, prot_train_env, prot_eval_env, prot_designer_lock = agent_thread()
        anta_lock, anta_train_env, anta_eval_env, anta_designer_lock = agent_thread()

        agent_envs = [prot_train_env, prot_eval_env, anta_train_env, anta_eval_env]
        agent_locks = [prot_lock, anta_lock]
        designer_locks = [prot_designer_lock, anta_designer_lock]

        from agent import make_agent_policy
        class WakupAgentsHook(StepHook):
            def __call__(self, env, agent, step):
                if env.done:
                    for lock in agent_locks:
                        # todo here, do agent_env.set(env params)
                        lock.release()  # let agents go
                    for lock in designer_locks:
                        lock.acquire()  # go to sleep

        experiments.train_agent_batch_with_evaluation(
            agent=make_agent_policy(args, obs_space, action_size, action_space, train_it=True),
            env=make_batch_env(test=False, discriminator=discriminator),
            eval_env=make_batch_env(test=True, discriminator=discriminator, is_evaluator=True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            max_episode_len=timestep_limit,
            use_tensorboard=True
        )


if __name__ == "__main__":
    main()