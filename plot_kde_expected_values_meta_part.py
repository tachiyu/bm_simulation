import os
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

savedir = f"results4/meta"
best_ts = {"SRAgent":{1: 6, 10: 19}, "QLearningAgent":{1: 4, 10: 6}}

os.makedirs(savedir, exist_ok=True)


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
def plot(paths, paths_b, paths_d,  paths_r):
    plt.clf()
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    evs = []
    labels = []

    # ランダムエージェントの値を点線で描画する
    plt.axhline(y=get_expected_value(distribution(paths_r[0])), color='gray', linestyle='--')

    distr = distribution(paths[0], meta=True)
    evs.append(get_expected_value(distr))
    labels.append(f"BM{bm_size1}\n learner")

    distr = distribution(paths_b[0])
    evs.append(get_expected_value(distr))
    labels.append(f"Begginer")

    distr = distribution(paths_d[0], meta=True)
    evs.append(get_expected_value(distr))
    labels.append(f"BM{bm_size2}'\n learner ")

    plt.xticks(range(len(evs)), labels)
     # 棒グラフを描画する
    plt.bar(range(len(evs)), evs)

     # グラフのタイトルと軸ラベルを追加する
    plt.title(f"BM{bm_size2} {agent}")
    plt.ylabel('Expected Log Travel Distance')

    savepath = f"{savedir}/evs_{agent}_BM{bm_size1}toBM{bm_size2}.png"
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

        
            