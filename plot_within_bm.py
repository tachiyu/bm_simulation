from pathlib import Path
import pickle
import re
from matplotlib import pyplot as plt
import numpy as np
import os

from analysis import Analysis

agent_types = ["SR"]
bm_sizes = [1, 3, 5, 10, 20]
table_widths = [1,2,3,4,5,6,7,8,9]
file_suffix = "_1_9"
show_error = True
savedir = f"results2/travel_dist/fig/within_bm/limit100000"
limit = 100000

os.makedirs(savedir, exist_ok=True)


def plot(paths, path_random, show_error, limit, savedir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for path in paths:
        with open(path, "rb") as f:
            tds = pickle.load(f)
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label="tableSize= "+re.search(r'table_width_(\d*)\D', path.name).group(1))
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)
    with open(path_random[0], "rb") as f:
        tds = pickle.load(f)
    q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
    ax.plot(q2, label=f"RandomAgent", c="gray", linestyle="--")
    if show_error:
        ax.fill_between(range(len(q2)), q1, q3, alpha=0.1, color="gray")

    ax.legend()
    ax.set_xlabel("trial")
    ax.set_ylabel("travel distance")
    ax.set_title(f"BM{bm_size}  {agent_type}")
    if limit is not None:
        ax.set_ylim(0, limit)
    plt.savefig(f"{savedir}/{agent_type}{file_suffix}.png")
    print(f"saved {savedir}/{agent_type}{file_suffix}.png")

if __name__ == "__main__":
    for bm_size in bm_sizes:
        savedir2 = f"{savedir}/BM{bm_size}"
        os.makedirs(savedir2, exist_ok=True)
        for agent_type in agent_types:
            paths = list(Path("results/travel_dist/pickle").glob(f"BM{bm_size}_*{agent_type}*a0.1*g0.9*greedy0.1*.pickle"))
            paths = [path for path in paths if int(re.search(r'table_width_(\d*)\D', path.name).group(1)) in table_widths] # from 1 to 9/
            pathr = list(Path("results/travel_dist/pickle").glob(f"BM{bm_size}_*RandomAgent*.pickle"))
            plot(paths, pathr, savedir=savedir2, limit=limit, show_error=show_error)
        
            