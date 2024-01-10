from pathlib import Path
import pickle
import re
from matplotlib import pyplot as plt
import numpy as np
import os

from analysis import Analysis

show_error = True
savedir = f"results3/travel_dist/fig/within_bm/meta_learning"
limits = [None, 100]
best_ts = {"SRAgent":{1: 6, 10: 20}, "QLearningAgent":{1: 4, 10: 6}}

os.makedirs(savedir, exist_ok=True)


def plot(paths, paths_b, paths_bb, paths_d, paths_db, paths_r, show_error, limit, savedir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for path, path_b, path_bb, path_d, path_db, path_r in zip(paths, paths_b, paths_bb, paths_d, paths_db, paths_r):
        with open(path, "rb") as f:
            tds = pickle.load(f)
        tds = [td[1] for td in tds]
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label=f"BM{bm_size1} learner")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)

        with open(path_d, "rb") as f:
            tds = pickle.load(f)
        tds = [td[1] for td in tds]
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label=f"BM{bm_size2} learner (matched table size)")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)

        with open(path_db, "rb") as f:
            tds = pickle.load(f)
        tds = [td[1] for td in tds]
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label=f"BM{bm_size2} learner (best table size)")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)

        with open(path_b, "rb") as f:
            tds = pickle.load(f)
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label=f"Begginer (matched table size)")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)

        with open(path_bb, "rb") as f:
            tds = pickle.load(f)
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label=f"Begginer (best table size)")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)

        with open(path_r, "rb") as f:
            tds = pickle.load(f)
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label="Random")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)

    ax.legend()
    ax.set_xlabel("Trial")
    ax.set_ylabel("Normarized travel distance")
    ax.set_title(f"BM{bm_size1} to BM{bm_size2}  {agent}")
    if limit is not None:
        ax.set_ylim(0, limit)
    savepath = f"{savedir}/{agent}_BM{bm_size1}toBM{bm_size2}.png" if limit is None else f"{savedir}/{agent}_BM{bm_size1}toBM{bm_size2}_limit{limit}.png"
    plt.savefig(savepath)
    print(f"{savepath} saved")

if __name__ == "__main__":
    savedir2 = f"{savedir}"
    os.makedirs(savedir2, exist_ok=True)
    for bm_sizes in [(1,10)]:
        bm_size1, bm_size2 = bm_sizes
        for agent in ['SRAgent', 'QLearningAgent']:
            for limit in limits:
                path = list(Path(f"results/travel_dist/pickle").glob(f"*BM{bm_size1}toBM{bm_size2}_{agent}_*"))
                print(path)
                path_b = list(Path(f"results/travel_dist/pickle").glob(f"BM{bm_size2}_{agent}_*a0.1*g0.9*greedy0.1*table_width_{best_ts[agent][bm_size1]}*"))
                print(path_b)
                path_bb = list(Path(f"results/travel_dist/pickle").glob(f"BM{bm_size2}_{agent}_*a0.1*g0.9*greedy0.1*table_width_{best_ts[agent][bm_size2]}*"))
                print(path_bb)
                path_d = list(Path(f"results/travel_dist/pickle").glob(f"*BM{bm_size2}toBM{bm_size2}_{agent}_*a0.1*g0.9*greedy0.1*table_width_{best_ts[agent][bm_size1]}*"))
                print(path_d)
                path_db = list(Path(f"results/travel_dist/pickle").glob(f"*BM{bm_size2}toBM{bm_size2}_{agent}_*a0.1*g0.9*greedy0.1*table_width_{best_ts[agent][bm_size2]}*"))
                print(path_db)
                path_r = list(Path(f"results/travel_dist/pickle").glob(f"BM{bm_size2}_*RandomAgent*"))
                print(path_r)
                plot(path, path_b, path_bb, path_d, path_db, path_r, savedir=savedir2, limit=limit, show_error=show_error)
                print()

        
            