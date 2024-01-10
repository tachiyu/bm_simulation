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

agents = ["SR", "Q"]
bm_sizes = [50]
savedir = f"results4/evs"
prefix = "ev_"
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

    table_sizes = [int(re.search(r'table_width_(\d*)\D', path.name).group(1)) for path in paths]
    distrs = [distribution(p) for p in paths]
    evs = [get_expected_value(d) for d in distrs]
    min_value = min(evs)
    min_index = evs.index(min_value)

    distrr = distribution(pathr[0])
    evr = get_expected_value(distrr)

    # 折れ線グラフの色を設定する
    line_color = 'blue'
    highlight_color = 'red'

    # ランダムエージェントの値を点線で描画する
    plt.axhline(y=evr, color='gray', linestyle='--')

    # 折れ線グラフを描画する
    plt.plot(table_sizes, evs, color=line_color, marker='o')
    # 最小値をハイライトする
    plt.plot(table_sizes[min_index], min_value, color=highlight_color, marker='o')

    # グラフのタイトルと軸ラベルを追加する
    plt.title(f"BM{bm_size} {agent}", fontsize=18)
    plt.xlabel('Table Size', fontsize=18)
    plt.ylabel('Expected Log Travel Distance', fontsize=18)

    labels = [l if l % 10 == 1 else "" for l in range(1, table_sizes[-1]+1)]
    # x軸にラベルを追加する
    plt.xticks(range(1, table_sizes[-1]+1), labels, fontsize=10)
    save_path = f"{savedir}/{prefix}{bm_size}_{agent}.png"

    # グラフを表示する
    plt.savefig(f"{save_path}")

for agent in agents:
    for bm_size in bm_sizes:
        plot()
