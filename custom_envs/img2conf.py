import argparse
import os.path

from PIL import Image
import glob
import numpy as np

from adversarial_env.adversarial_minigrid import AdversarialEnv

DEBUG = False
def print_debug(*str):
    if DEBUG:
        print(str)

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str, default=".", nargs='?')
parser.add_argument("output_dir", type=str, default=".", nargs='?')
parser = parser.parse_args()

AGENT = (245, 0, 0)  # red
GOAL = (0, 245, 0)  # green
WALL = (0, 0, 245)  # blue

name_conf_pairs = []
for filename in glob.glob(f'{parser.input_dir}/*.png'):
    img = Image.open(filename)
    # if RGBA, only keep 3 channels
    arr = np.array(img)[:, :, :3]
    print_debug("Should be 3 channels:\t\t", arr.shape)
    arr = arr[1:-1, 1:-1,:]
    # flatten the 2d array into one
    flat = arr.reshape(-1, 3)
    print_debug("Should be of shape (h * w, 3):\t", flat.shape)

    # get locations
    goal_locs = np.where(np.all(flat >= GOAL, axis=-1))[0]
    wall_locs = np.where(np.all(flat >= WALL, axis=-1))[0]
    agent_locs = np.where(np.all(flat >= AGENT, axis=-1))[0]

    print_debug("\nParsed pixels:")
    print_debug("Goal loc:\t", goal_locs)
    print_debug("Agent loc:\t", agent_locs)
    print_debug("Wall loc:\t", wall_locs)

    # create adv env step list. Order is: Goal, Agent, Wall, Wall ...
    step_list = [goal_locs[0], agent_locs[0]] + wall_locs.tolist()
    env = AdversarialEnv(size=arr.shape[0]+2)   # +2 because outer wall
    env.reset()
    for step in step_list:
        env.step(step)

    if DEBUG:
        img = img.resize((300,300))
        img.show(title="original")

        img = Image.fromarray(env.render(mode="inhuman, lol"))
        img = img.resize((300,300))
        img.show(title="conf")

    conf = env.get_env_configuration()
    conf.name = os.path.basename(filename).split(".")[0]
    name_conf_pairs.append((filename, conf))

import pickle
for filename, conf in name_conf_pairs:
    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(conf, f, protocol=pickle.HIGHEST_PROTOCOL)

