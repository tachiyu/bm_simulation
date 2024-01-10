from pathlib import Path
import pickle
import re
from matplotlib import cm, pyplot as plt
import numpy as np
import os
import seaborn as sns

from analysis import Analysis

show_error = True
savedir = f"results4/meta"
best_ts = {"SRAgent":{1: 6, 10: 19}, "QLearningAgent":{1: 4, 10: 6}}
plt.rcParams.update({'font.size': 16})

os.makedirs(savedir, exist_ok=True)

def distribution(path, meta=False):
    with open(path, "rb") as f:
        travel_dists = pickle.load(f)
    if meta:
        travel_dists = [td[1] for td in travel_dists]
    distr = []
    for travel_dist in travel_dists:
        distr += travel_dist
    return distr

# plot histograms of first and last 5 trials, x is lo
def plot(paths, paths_b, paths_d, paths_r):

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cmap = plt.cm.turbo

    def p(distr, color, label, random=False):
        if random:
            sns.kdeplot(distr, label=label, color=color, log_scale=True, linestyle="--")
        else:
            sns.kdeplot(distr, label=label, color=color, log_scale=True)
        print("plotting", label)

    distr = distribution(paths[0], meta=True)
    p(distr, cmap(0/3), f"BM{bm_size1} learner")

    distr = distribution(paths_b[0])
    p(distr, cmap(1/3), f"Begginer")

    distr = distribution(paths_d[0], meta=True)
    p(distr, cmap(2/3), f"BM{bm_size2}' learner")

    distr = distribution(paths_r[0])
    p(distr, "gray", f"Random", random=True)

    # 凡例をプロットに追加
    plt.legend()
    ax.set_xlabel("Travel distance")
    ax.set_ylabel("Frequency")
    ax.set_title(f"BM{bm_size2} {agent}")
    savepath = f"{savedir}/kde_{agent}_BM{bm_size1}toBM{bm_size2}.png"
    plt.savefig(savepath)
    print(f"{savepath} saved")

if __name__ == "__main__":
    for bm_sizes in [(1,10), (10,1)]:
        bm_size1, bm_size2 = bm_sizes
        for agent in ['SRAgent', 'QLearningAgent']:
            path = list(Path(f"results/travel_dist/pickle").glob(f"*BM{bm_size1}toBM{bm_size2}_{agent}_*"))
            print(path)
            path_b = list(Path(f"results/travel_dist/pickle").glob(f"BM{bm_size2}_{agent}_*a0.1*g0.9*greedy0.1*table_width_{best_ts[agent][bm_size2]}*"))
            print(path_b)
            path_d = list(Path(f"results/travel_dist/pickle").glob(f"*BM{bm_size2}toBM{bm_size2}_{agent}_*a0.1*g0.9*greedy0.1*table_width_{best_ts[agent][bm_size2]}*"))
            print(path_d)
            path_r = list(Path(f"results/travel_dist/pickle").glob(f"BM{bm_size2}_*RandomAgent*"))
            print(path_r)
            plot(path, path_b, path_d, path_r)
            print()

        
            