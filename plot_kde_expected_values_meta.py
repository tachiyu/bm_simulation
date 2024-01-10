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

savedir = f"results3/travel_dist/fig/within_bm/meta_learning"
best_ts = {"SRAgent":{1: 6, 10: 20}, "QLearningAgent":{1: 4, 10: 6}}

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
def plot(paths, paths_b, paths_bb, paths_d, paths_db, paths_r):
    plt.clf()
    plt.subplots_adjust(bottom = 0.25)
    evs = []
    labels = []

    # ランダムエージェントの値を点線で描画する
    plt.axhline(y=get_expected_value(distribution(paths_r[0])), color='gray', linestyle='--')

    distr = distribution(paths[0], meta=True)
    evs.append(get_expected_value(distr))
    labels.append(f"BM{bm_size1} learner")

    distr = distribution(paths_d[0], meta=True)
    evs.append(get_expected_value(distr))
    labels.append(f"BM{bm_size2}' learner\n(matched table size)")
    
    distr = distribution(paths_db[0], meta=True)
    evs.append(get_expected_value(distr))
    labels.append(f"BM{bm_size2}' learner\n(2nd)")

    distr = distribution(paths_b[0])
    evs.append(get_expected_value(distr))
    labels.append(f"Begginer\n")

    distr = distribution(paths_bb[0])
    evs.append(get_expected_value(distr))
    labels.append(f"Begginer\n(best table size)")

    plt.xticks(range(len(evs)), labels, rotation=30)
     # 棒グラフを描画する
    plt.bar(range(len(evs)), evs)

     # グラフのタイトルと軸ラベルを追加する
    plt.title(f"BM{bm_size1} to BM{bm_size2} {agent}")
    plt.ylabel('Expected Log Value')

    savepath = f"{savedir}/ExpScore_{agent}_BM{bm_size1}toBM{bm_size2}.png"
    plt.savefig(savepath)
    print(f"{savepath} saved")

if __name__ == "__main__":
    for bm_sizes in [(10,1), (1,10)]:
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

        
            