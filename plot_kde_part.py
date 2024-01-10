from pathlib import Path
import pickle
import re

from matplotlib import cm, lines, pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import sort_file_with_tSiz

window=3

def get_paths():
    paths = list(Path("results/travel_dist/pickle").glob(f"BM{bm_size}_*{agent}*a0.1*g0.9*greedy0.1*.pickle"))
    paths = sort_file_with_tSiz(paths)
    pathr = list(Path("results/travel_dist/pickle").glob(f"BM{bm_size}_*RandomAgent*a0.1*g0.9*greedy0.1*.pickle"))
    return paths, pathr

def out_dist(pash):
    f_distr, r_distr = distribution(pash)
    d = pd.DataFrame({"first":f_distr, "last":r_distr})
    d.to_csv(f"{agent}_BM{bm_size}_first_last.csv")

def distribution(path):
    with open(path, "rb") as f:
        travel_dists = pickle.load(f)
    first_distr = []
    for travel_dist in travel_dists:
        first_distr += travel_dist[:window]
    last_distr = []
    for travel_dist in travel_dists:
        last_distr += travel_dist[-window:]
    return first_distr, last_distr

# plot histograms of first and last 5 trials, x is lo
def plot():
    paths, pathr = get_paths()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # カラーマップから色を取得
    n_colors = len(paths)
    colors = cm.jet(np.linspace(0, 1, n_colors))
    for i, path in enumerate(paths):
        first_distr, last_distr = distribution(path)
        table_width = int(re.search(r'table_width_(\d*)\D', path.name).group(1))
        sns.kdeplot(first_distr, color=colors[i], linestyle="--", bw_adjust=0.5, log_scale=True)
        sns.kdeplot(last_distr, label=f'tSiz={table_width}', color=colors[i], bw_adjust=0.5, log_scale=True)
        print("plotting", path.name)
    first_distr_r, last_distr_r = distribution(pathr[0])
    sns.kdeplot(first_distr_r, color="gray",label='Random', bw_adjust=0.5, log_scale=True)
    print("plotting", pathr[0].name)

    # カスタム凡例エントリの作成
    legend_line_a = lines.Line2D([], [], color='black', marker='', linestyle='--', label=f'First {window} trials')
    legend_line_b = lines.Line2D([], [], color='black', marker='', linestyle='-', label=f'Last {window} trials')

    # 既存のプロットから凡例エントリを取得
    existing_legend = plt.gca().get_legend_handles_labels()

    # 既存の凡例エントリにカスタムエントリを追加
    handles, labels = existing_legend
    handles.extend([legend_line_a, legend_line_b])
    labels.extend([f'First {window} trials', f'Last {window} trials'])

    # 凡例をプロットに追加
    plt.legend(handles=handles, labels=labels)
    ax.set_xlabel("travel distance")
    ax.set_ylabel("frequency")
    ax.set_title(f"BM{bm_size} {agent}")
    plt.savefig(f"{save_path}")
    print(f"{save_path} saved")

for agent in ["Q", "SR"]:
    for bm_size in [1,3,5,10,20]:
        save_path = Path(f"results3/kde/{agent}_BM{bm_size}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plot()
