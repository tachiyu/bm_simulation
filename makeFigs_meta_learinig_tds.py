from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import numpy as np
import os

show_error = False
savedir = f"results/figs/meta_learning"
best_ts = {"SRAgent":{1: 6, 10: 19}, "QLearningAgent":{1: 4, 10: 6}}
agent = "SRAgent"
params = {"QLearningAgent": {1:[None,50,1000],10:[None,200,1000,10000,100000]}, "SRAgent": {1:[None, 50],10:[None,5000]}}
prefix = "tds_"

os.makedirs(savedir, exist_ok=True)
plt.rcParams.update({'font.size': 16})

def plot(paths, paths_b, paths_d, paths_r, show_error, limit, savedir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for path, path_b, path_d,path_r in zip(paths, paths_b, paths_d, paths_r):
        with open(path, "rb") as f:
            tds = pickle.load(f)
        tds = [td[1] for td in tds]
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label=f"BM{bm_size1} learner")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)

        with open(path_b, "rb") as f:
            tds = pickle.load(f)
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label=f"Begginer")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)

        with open(path_d, "rb") as f:
            tds = pickle.load(f)
        tds = [td[1] for td in tds]
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label=f"BM{bm_size2}' learner")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)

        with open(path_r, "rb") as f:
            tds = pickle.load(f)
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2, label="Random", linestyle="--")
        if show_error:
            ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)

    ax.legend()
    ax.set_xlabel("Trial")
    ax.set_ylabel("Travel distance")
    ax.set_title(f"BM{bm_size2}  {agent}")
    if limit is not None:
        ax.set_ylim(0, limit)
    savepath = f"{savedir}/{prefix}{agent}_BM{bm_size1}toBM{bm_size2}.png" if limit is None else f"{savedir}/tds_{agent}_BM{bm_size1}toBM{bm_size2}_limit{limit}.png"
    plt.savefig(savepath)
    print(f"{savepath} saved")

if __name__ == "__main__":
    savedir2 = f"{savedir}"
    os.makedirs(savedir2, exist_ok=True)
    for bm_size1, bm_size2 in [(1,10), (10,1)]:
        for agent in params.keys():
            for limit in params[agent][bm_size2]:
                path = list(Path(f"results/pickles/travel_dists").glob(f"*BM{bm_size1}toBM{bm_size2}_{agent}_*"))
                print(path)
                path_b = list(Path(f"results/pickles/travel_dists").glob(f"BM{bm_size2}_{agent}_*a0.1*g0.9*greedy0.1*table_width_{best_ts[agent][bm_size2]}*"))
                print(path_b)
                path_d = list(Path(f"results/pickles/travel_dists").glob(f"*BM{bm_size2}toBM{bm_size2}_{agent}_*a0.1*g0.9*greedy0.1*table_width_{best_ts[agent][bm_size2]}*"))
                print(path_d)
                path_r = list(Path(f"results/pickles/travel_dists").glob(f"BM{bm_size2}_*RandomAgent*"))
                print(path_r)
                plot(path, path_b, path_d, path_r, savedir=savedir2, limit=limit, show_error=show_error)
                print()

        
            