from pathlib import Path
import pickle
import re

from matplotlib import cm, lines, pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utils import sort_file_with_tSiz

from scipy.integrate import simpson
from scipy.stats import gaussian_kde

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

def get_expected_value(distr):
    # 対数スケールでのデータセット
    log_data = np.log(distr)
    # カーネル密度推定オブジェクトを作成
    kde = gaussian_kde(log_data)
    # 密度関数の定義される範囲を設定
    grid = np.linspace(min(log_data), max(log_data), 1000)
    # 密度関数を評価
    kde_values = kde(grid)
    # 密度関数による重み付けで期待値（平均）を計算
    expected_value = simpson(kde_values * grid, grid)
    return expected_value

# plot histograms of first and last 5 trials, x is lo
def plot():
    plt.clf()
    paths, pathr = get_paths()

    distrs = [distribution(p) for p in paths]
    evs = [get_expected_value(d) for d in distrs]
    min_value = min(evs)
    min_index = evs.index(min_value)

    distrr = distribution(pathr[0])
    evr = get_expected_value(distrr)
    evs.append(evr)

    # 棒グラフの色を設定する
    colors = ['blue' if i != min_index else 'red' for i in range(len(evs))]
    colors.append('gray')

    # 棒グラフを描画する
    plt.bar(range(len(evs)), evs, color=colors)

    # グラフのタイトルと軸ラベルを追加する
    plt.title(f"BM{bm_size} {agent}")
    plt.xlabel('Table Size')
    plt.ylabel('Expected Score')

    labels = [int(re.search(r'table_width_(\d*)\D', path.name).group(1)) for path in paths]
    labels.append("Rand")
    # x軸にラベルを追加する
    plt.xticks(range(len(evs)), labels)

    # グラフを表示する
    plt.savefig(f"{save_path}")

for agent in ["Q", "SR"]:
    for bm_size in [1,3,5,10,20]:
        save_path = Path(f"results2/kde_expected_values/{agent}_BM{bm_size}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plot()
