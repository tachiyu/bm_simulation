from pathlib import Path
import pickle
import re

from matplotlib import cm, pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

params = {"QLearningAgent": [50], "SRAgent": [50]}
savedir = f"results/figs/kde_all"
file_prefix = ""

def get_paths(bm_size, agent):
    paths = list(Path("results/pickles/travel_dists").glob(f"BM{bm_size}_*{agent}*a0.1*g0.9*greedy0.1*.pickle"))
    pathr = list(Path("results/pickles/travel_dists").glob(f"BM{bm_size}_*RandomAgent*a0.1*g0.9*greedy0.1*.pickle"))
    return paths, pathr

def distribution(path):
    with open(path, "rb") as f:
        travel_dists = pickle.load(f)
    distr = []
    for travel_dist in travel_dists:
        distr += travel_dist
    return distr

# plot histograms of first and last 5 trials, x is lo
def plot(bm_size, agent):
    paths, pathr = get_paths(bm_size, agent)
    plt.rcParams.update({'font.size': 16})
    cmap = plt.cm.turbo
        
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    paths = sorted(paths, key=lambda x: int(re.search(r'table_width_(\d+)', x.name).group(1)))
    table_sizes = [int(re.search(r'table_width_(\d*)\D', path.name).group(1)) for path in paths]
     # カラーマップの範囲を設定
    norm = Normalize(vmin=1, vmax=table_sizes[-1]+1)

    for path in paths:
        distr = distribution(path)
        table_width = int(re.search(r'table_width_(\d*)\D', path.name).group(1))
        sns.kdeplot(distr, log_scale=True, c=cmap(norm(table_width)))
        print("plotting", path.name)
    distr_r = distribution(pathr[0])
    sns.kdeplot(distr_r, color="gray",label='Random', linestyle="--", log_scale=True)
    print("plotting", pathr[0].name)

    # 凡例をプロットに追加
    plt.legend()
    ax.set_xlabel("Travel Distance")
    ax.set_ylabel("Density")
    ax.set_title(f"BM{bm_size} {agent}")
    # カラーバーの作成
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Table Size')  # カラーバーのラベル
    path = f"{savedir}/{file_prefix}{bm_size}_{agent}.png"
    plt.savefig(f"{path}")
    print(f"{path} saved")

for agent_type in params.keys():
    for bm_size in params[agent_type]:
        plot(bm_size, agent_type)
