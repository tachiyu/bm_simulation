from pathlib import Path
import pickle
import re
from matplotlib import pyplot as plt
import numpy as np
import os

savedir = "results/figs/h_params"
bm_sizes = [1, 3, 5, 10]
table_widths = [5, 10, 30]
agent_types = ["Q", "SR"]
suffix = "t"
show_error = False
limit = None


def plot(paths, path_r, suffix, show_error, limit):
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
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
        ax.plot(q2, label=f"Random", c="gray", linestyle="--")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2, color="gray")
    ax.legend()
    ax.set_xlabel("Trial")
    ax.set_ylabel("Travel Distance")
    ax.set_title(f"BM{bm_size}  {agent_type}  tableSize={table_width}")
    if limit is not None:
        ax.set_ylim(0, limit)
    plt.savefig(f"{savedir}/{suffix}BM{bm_size}_{agent_type}_tSiz{table_width}.png")
    plt.close()

if __name__ == "__main__":
    os.makedirs(savedir, exist_ok=True)
    plt.rcParams.update({'font.size': 16})
    for bm_size in bm_sizes:
        for table_width in table_widths:
            for agent_type in agent_types:
                paths = Path("results/pickles/travel_dists").glob(f"BM{bm_size}_*{agent_type}*table_width_{table_width}*.pickle")
                path_r = Path("results/pickles/travel_dists").glob(f"BM{bm_size}_*RandomAgent*.pickle")
                plot(paths, path_r, suffix, show_error, limit)
        
            