from pathlib import Path
import pickle
import re
from matplotlib import pyplot as plt
import numpy as np

from analysis import Analysis

def plot(paths, suffix="", show_error=True, limit=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for path in paths:
        with open(path, "rb") as f:
            tds = pickle.load(f)
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        agent_type = re.search(r'BM\d+_(.+?)_', path.name).group(1)
        table_with = re.search(r'table_width_(\d+)', path.name).group(1)
        alpha = re.search(r'a\d+\.\d+', path.name).group(0)
        gamma = re.search(r'g\d+\.\d+', path.name).group(0)
        epsilon = 'e' + re.search(r'greedy(\d+\.\d+)', path.name).group(1) if re.search(r'greedy\d+\.\d+', path.name) else "e0.1"
        ax.plot(q2, label=f"{agent_type}_{table_with} (n={len(tds)})")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.5)
    ax.legend()
    if limit is not None:
        ax.set_ylim(0, limit)
    ax.set_title(f"BM{bm_size}")
    plt.savefig(f"results/travel_dist/fig/BM{bm_size}{suffix}.png")

if __name__ == "__main__":
    bm_sizes = ["20"]
    for bm_size in bm_sizes:
        paths = Path("results/travel_dist/pickle").glob(f"BM{bm_size}_*.pickle")
        # paths = [path for path in paths if re.search(r'table_width_\d\D', path.name)]
        # paths = [path for path in paths if re.search(r'Q', path.name) and not re.search(r'greedy\d+', path.name) and re.search(r'a0.1', path.name) and re.search(r'g0.9', path.name)]
        paths = [path for path in paths if not re.search(r'table_width_3\D', path.name)]
        plot(paths, suffix="", show_error=True)
        
            