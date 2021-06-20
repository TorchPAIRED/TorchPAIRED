import argparse
import logging
import sys

from pfrl import experiments

from env_registrations import env_registrations


def get_args():
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
        default="minigrid",
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
        "--render", action="store_true", default=False, help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument(
        "--monitor", action="store_true", default=True, help="Wrap env with Monitor to write videos."
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
        default=300,
        # https://github.com/pfnet/pfrl/blob/44bf2e483f5a2f30be7fd062545de306247699a1/examples/gym/train_reinforce_gym.py#L84
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

    args.env = env_registrations[args.env]

    logging.basicConfig(level=args.log_level)

    args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    print("Output files are saved in {}".format(args.outdir))

    with open(args.outdir + "/args.txt", "w") as f:
        f.write("\n".join(str(args).split(",")))

    return args