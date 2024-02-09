from pathlib import Path
import pickle
import os

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

from scipy.integrate import simpson
from scipy.stats import gaussian_kde

savedir = f"results/figs/example"
file_prefix = ""
paths = [
    Path("results/pickles/travel_dists/BM1_SRAgent_no_habituation_n30_ne36_a0.1_g0.5_pe_greedy0.1_table_width_6_trajOFF.pickle"),
    Path("results/pickles/travel_dists/BM1_SRAgent_no_habituation_n30_ne36_a0.1_g0.5_pe_greedy0.1_table_width_30_trajOFF.pickle"),
]

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
    plt.rcParams.update({'font.size': 16})

    # learning curve
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    for path in paths:
        with open(path, "rb") as f:
            tds = pickle.load(f)
        q1, q2, q3 = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*tds)])
        ax.plot(q2)
        ax.fill_between(range(len(q2)), q1, q3, alpha=0.2)
    path = f"{savedir}/{file_prefix}learning_curve.png"
    plt.savefig(f"{path}")
    print(f"{path} saved")
    plt.close()

    # kde all
    _, ax = plt.subplots(1, 1, figsize=(10, 6))
    for path in paths:
        distr = distribution(path)
        sns.kdeplot(distr, log_scale=True)
        print("plotting", path.name)
    path = f"{savedir}/{file_prefix}kde_all.png"
    plt.savefig(f"{path}")
    print(f"{path} saved")

    # print evs
    for path in paths:
        distr = distribution(path)
        ev = get_expected_value(distr)
        print(f"{path.name} ev: {ev}")

os.makedirs(savedir, exist_ok=True)
plot()
