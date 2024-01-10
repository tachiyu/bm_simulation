from pathlib import Path
import pickle
import re

from matplotlib import cm, lines, pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import simpson
from scipy.stats import gaussian_kde


savedir = "to_progress_report"

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
    p1 = r"results/travel_dist/pickle/BM1_QLearningAgent_no_habituation_n30_ne36_a0.1_g0.9_pe_greedy0.1_table_width_3_trajOFF.pickle"
    p2 = r"results/travel_dist/pickle/BM1_QLearningAgent_no_habituation_n30_ne36_a0.1_g0.9_pe_greedy0.1_table_width_30.pickle"

    distr = distribution(p1)
    sns.kdeplot(distr, label=f'table size=3', bw_adjust=0.5, log_scale=True, color="red")
    distr = distribution(p2)
    sns.kdeplot(distr, label=f'table size=30', bw_adjust=0.5, log_scale=True, color="blue")

    # 凡例をプロットに追加
    plt.legend(fontsize=14)
    plt.xlabel("Travel Distance (log scale)", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.savefig(f"{savedir}/sample_kde.png")

def plot2():
    plt.rcParams["font.size"] = 14
    p1 = r"results/travel_dist/pickle/BM1_QLearningAgent_no_habituation_n30_ne36_a0.1_g0.9_pe_greedy0.1_table_width_3_trajOFF.pickle"
    p2 = r"results/travel_dist/pickle/BM1_QLearningAgent_no_habituation_n30_ne36_a0.1_g0.9_pe_greedy0.1_table_width_30.pickle"

    evs = []
    labesl = []
    distr = distribution(p1)
    ev = get_expected_value(distr)
    evs.append(ev)
    labesl.append(f'3')

    distr = distribution(p2)
    ev = get_expected_value(distr)
    evs.append(ev)
    labesl.append(f'30')

    # 凡例をプロットに追加
    bar = plt.bar(range(len(evs)), evs, color="red")
    plt.xticks(range(len(evs)), labesl)
    bar[1].set_color("blue")

    plt.xlabel("Table Size")
    plt.ylabel("Expected Log Travel Distance")
    plt.savefig(f"{savedir}/sample_eltd.png")

plot()
plot2()

