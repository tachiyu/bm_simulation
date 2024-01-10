from pathlib import Path
import pickle
import re
from matplotlib import pyplot as plt
import numpy as np
import os

from analysis import Analysis

params = {"SRAgent": {1: [1,4,6,8,10,20,30], 3:[1,5,8,11,14,20,30], 5:[1,3,6,10,15,20,30], 10:[1,10,15,20,25,30,35,40], 20:[1,20,25,30,35,40,45,50]}, "QLearningAgent": {1: [2,3,4,5,10,20,30], 3: [2,3,4,5,10,20,30], 5: [3,5,6,8,10,20,30], 10: [3,5,6,8,10,20,30], 20:[4,6,10,14,20,30]}}

file_prefix = "travel_distance_"

show_error = False
savedir = f"results4/travel_distance"
bm_size = 10
limit =500
file_suffix = f"_limit{limit}" if limit is not None else ""

os.makedirs(savedir, exist_ok=True)


def plot(paths, path_random, show_error, limit, savedir):
    plt.rcParams.update({'font.size': 16})
    cmap = plt.cm.turbo
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    paths = sorted(paths, key=lambda x: int(re.search(r'table_width_(\d+)', x.name).group(1)))
    for i, path in enumerate(paths):
        with open(path, "rb") as f:
            tds = pickle.load(f)
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label="tSiz= "+re.search(r'table_width_(\d*)\D', path.name).group(1), c=cmap(i/len(paths)))
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2, c=cmap(i/len(paths)))
    with open(path_random[0], "rb") as f:
        tds = pickle.load(f)
    q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
    ax.plot(q2, label=f"Random", c="gray", linestyle="--")
    if show_error:
        ax.fill_between(range(len(q2)), q1, q3, alpha=0.1, color="gray")

    ax.legend()
    ax.set_xlabel("Trial")
    ax.set_ylabel("Travel Distance")
    ax.set_title(f"BM{bm_size}  {agent_type}")
    if limit is not None:
        ax.set_ylim(0, limit)
    path = f"{savedir}/{file_prefix}{bm_size}_{agent_type}{file_suffix}.png"
    plt.savefig(path)
    print(f"saved {path}")

if __name__ == "__main__":
    for agent_type in params.keys():
        table_widths = params[agent_type][bm_size]
        paths = list(Path("results/travel_dist/pickle").glob(f"BM{bm_size}_*{agent_type}*a0.1*g0.9*greedy0.1*.pickle"))
        paths = [path for path in paths if int(re.search(r'table_width_(\d*)\D', path.name).group(1)) in table_widths] # from 1 to 9/
        pathr = list(Path("results/travel_dist/pickle").glob(f"BM{bm_size}_*RandomAgent*.pickle"))
        plot(paths, pathr, savedir=savedir, limit=limit, show_error=show_error)
        
            