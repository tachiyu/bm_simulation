import os
import pickle
import re

from agents.agent import Agent
from envs import BM

def save_bm(bm, name):
    path = f"results/pickles/{name}.pickle"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm, f)

def sort_file_with_tSiz(paths):
    def get_tSiz(path):
        return int(re.search(r'table_width_(\d*)', path.name).group(1))
    return sorted(paths, key=get_tSiz)