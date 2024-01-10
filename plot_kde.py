from pathlib import Path
import pickle
import re

from matplotlib import cm, lines, pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import sort_file_with_tSiz
# params = {"QLearningAgent": {1: [2,3,4,5,10,20,30], 3: [2,3,4,5,10,20,30], 5: [3,5,6,8,10,20,30], 10: [3,5,6,8,10,20,30], 20:[4,6,10,14,20,30]},
#            }
# params = {"SRAgent": {1: [1,4, 6, 8,10,20,30], 3: [1,5,8,11,14,20,30], 5: [1,3, 6, 10,15,20,30], 10: [1,10,15,20,25,30, 35,40], 20:[1,20,25,30,35, 40, 45, 50]}}
params = {"QLearningAgent": {1: [1], 3: [1], 5: [1, 2], 10: [1,2], 20:[1,2,3]}}
params = {"SRAgent": {10: [3,4,5], 20:[3,4,5,6]}}



savedir = f"results4/kde_worse"
file_prefix = "kde_"

def get_paths(bm_size, agent, table_widths):
    paths = list(Path("results/travel_dist/pickle").glob(f"BM{bm_size}_*{agent}*a0.1*g0.9*greedy0.1*.pickle"))
    paths = [path for path in paths if int(re.search(r'table_width_(\d*)\D', path.name).group(1)) in table_widths]
    pathr = list(Path("results/travel_dist/pickle").glob(f"BM{bm_size}_*RandomAgent*a0.1*g0.9*greedy0.1*.pickle"))
    return paths, pathr

def distribution(path):
    with open(path, "rb") as f:
        travel_dists = pickle.load(f)
    distr = []
    for travel_dist in travel_dists:
        distr += travel_dist
    return distr

# plot histograms of first and last 5 trials, x is lo
def plot(bm_size, agent, table_widths):
    paths, pathr = get_paths(bm_size, agent, table_widths)
    plt.rcParams.update({'font.size': 16})
    cmap = plt.cm.turbo
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    paths = sorted(paths, key=lambda x: int(re.search(r'table_width_(\d+)', x.name).group(1)))
    for i, path in enumerate(paths):
        distr = distribution(path)
        table_width = int(re.search(r'table_width_(\d*)\D', path.name).group(1))
        sns.kdeplot(distr, label=f'tSiz={table_width}', log_scale=True, c=cmap(i/len(paths)))
        print("plotting", path.name)
    distr_r = distribution(pathr[0])
    sns.kdeplot(distr_r, color="gray",label='Random', linestyle="--", log_scale=True)
    print("plotting", pathr[0].name)

    # 凡例をプロットに追加
    plt.legend()
    ax.set_xlabel("Travel Distance")
    ax.set_ylabel("Density")
    ax.set_title(f"BM{bm_size} {agent}")
    path = f"{savedir}/{file_prefix}{bm_size}_{agent}.png"
    plt.savefig(f"{path}")
    print(f"{path} saved")

for agent_type in params.keys():
    for bm_size in params[agent_type].keys():
        table_widths = params[agent_type][bm_size]
        plot(bm_size, agent_type, table_widths)
