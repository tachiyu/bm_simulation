from pathlib import Path
import pickle
import re
from matplotlib import pyplot as plt
import numpy as np
import os

from analysis import Analysis

savedir = "results2/travel_dist/fig/h_params_test/plot_a0.2"
os.makedirs(savedir, exist_ok=True)

bm_sizes = [1, 3, 5, 10]
table_widths = [5, 10, 30]
agent_types = ["Q", "SR"]

def plot(paths, path_r, suffix="", show_error=True, limit=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for path in paths:
        with open(path, "rb") as f:
            tds = pickle.load(f)
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        alpha = re.search(r'a(\d+\.\d+)', path.name).group(1)
        gamma = re.search(r'g(\d+\.\d+)', path.name).group(1)
        epsilon = re.search(r'greedy(\d+\.\d+)', path.name).group(1)
        ax.plot(q2, label=f"α={alpha}, γ={gamma}, ε={epsilon}")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)
    for path in path_r:
        with open(path, "rb") as f:
            tds = pickle.load(f)
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label=f"RandomAgent", c="gray", linestyle="--")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2, color="gray")
    ax.legend()
    ax.set_xlabel("trial")
    ax.set_ylabel("travel distance")
    ax.set_title(f"BM{bm_size}  {agent_type}  tableSize={table_width}")
    if limit is not None:
        ax.set_ylim(0, limit)
    plt.savefig(f"{savedir}/BM{bm_size}_{agent_type}_tSiz{table_width}.png")

if __name__ == "__main__":
    for bm_size in bm_sizes:
        for table_width in table_widths:
            for agent_type in agent_types:
                paths = Path("results/travel_dist/pickle").glob(f"BM{bm_size}_*{agent_type}*table_width_{table_width}*.pickle")
                path_r = Path("results/travel_dist/pickle").glob(f"BM{bm_size}_*RandomAgent*.pickle")
                plot(paths, path_r, suffix="", show_error=True)
        
            