from pathlib import Path
import pickle
import re
from matplotlib import cm, pyplot as plt
import numpy as np
import os
import seaborn as sns

from analysis import Analysis

show_error = True
savedir = f"results3/travel_dist/fig/within_bm/meta_learning"
best_ts = {"SRAgent":{1: 6, 10: 20}, "QLearningAgent":{1: 4, 10: 6}}

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
def plot(paths, paths_b, paths_bb, paths_d, paths_db, paths_r):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # カラーマップから色を取得
    n_colors = 5
    colors = cm.jet(np.linspace(0, 1, n_colors))

    def p(distr, color, label):
        sns.kdeplot(distr, label=label, color=color, bw_adjust=0.5, log_scale=True)
        print("plotting", label)

    distr = distribution(paths[0], meta=True)
    p(distr, colors[0], f"BM{bm_size1} learner")

    distr = distribution(paths_d[0], meta=True)
    p(distr, colors[1], f"BM{bm_size2} learner (matched table size)")
    
    distr = distribution(paths_db[0], meta=True)
    p(distr, colors[2], f"BM{bm_size2} learner (best table size)")

    distr = distribution(paths_b[0])
    p(distr, colors[3], f"Begginer (matched table size)")

    distr = distribution(paths_bb[0])
    p(distr, colors[4], f"Begginer (best table size)")

    distr = distribution(paths_r[0])
    p(distr, "gray", f"Random")

    # 凡例をプロットに追加
    plt.legend()
    ax.set_xlabel("Travel distance")
    ax.set_ylabel("Frequency")
    ax.set_title(f"BM{bm_size1} to BM{bm_size2} {agent}")
    savepath = f"{savedir}/KDE_all_{agent}_BM{bm_size1}toBM{bm_size2}.png"
    plt.savefig(savepath)
    print(f"{savepath} saved")

if __name__ == "__main__":
    for bm_sizes in [(10,1)]:
        bm_size1, bm_size2 = bm_sizes
        for agent in ['SRAgent', 'QLearningAgent']:
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
            plot(path, path_b, path_bb, path_d, path_db, path_r)
            print()

        
            