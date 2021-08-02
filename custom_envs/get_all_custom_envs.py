import argparse
import glob
import pickle

from adversarial_env.settable_minigrid import SettableMinigrid

def get_all_custom_envs(input_dir):
    envs = []
    for filename in glob.glob(f'{input_dir}/*.pkl'):
        with open(filename, 'rb') as handle:
            conf = pickle.load(handle)
        env = SettableMinigrid(conf.size)
        env.env_name = conf.name
        env.set(conf)
        envs.append(env)
    return envs

if __name__ == "__main__":
    get_all_custom_envs(".")