from pathlib import Path
import pickle
import re

from matplotlib import cm, lines, pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import sort_file_with_tSiz

def get_paths():
    paths = list(Path("results/travel_dist/pickle").glob(f"BM{bm_size}_*{agent}*a0.1*g0.9*greedy0.1*.pickle"))
    paths = sort_file_with_tSiz(paths)
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
def plot():
    paths, pathr = get_paths()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # カラーマップから色を取得
    n_colors = len(paths)
    colors = cm.jet(np.linspace(0, 1, n_colors))
    for i, path in enumerate(paths):
        distr = distribution(path)
        table_width = int(re.search(r'table_width_(\d*)\D', path.name).group(1))
        sns.kdeplot(distr, label=f'tSiz={table_width}', color=colors[i], bw_adjust=0.5, log_scale=True)
        print("plotting", path.name)
    distr_r = distribution(pathr[0])
    sns.kdeplot(distr_r, color="gray",label='Random', bw_adjust=0.5, log_scale=True)
    print("plotting", pathr[0].name)

    # 凡例をプロットに追加
    plt.legend()
    ax.set_xlabel("travel distance")
    ax.set_ylabel("frequency")
    ax.set_title(f"BM{bm_size} {agent}")
    plt.savefig(f"{save_path}")
    print(f"{save_path} saved")

for agent in ["Q", "SR"]:
    for bm_size in [1,3,5,10,20]:
        save_path = Path(f"results2/kde_all/{agent}_BM{bm_size}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plot()
