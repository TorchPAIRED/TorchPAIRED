import argparse
import glob
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str, default=".", nargs='?')
parser = parser.parse_args()
subprocess.call(f"cd {parser.input_dir} && rm -rf *.pkl", shell=True)