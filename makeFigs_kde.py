from pathlib import Path
import pickle
import re

from matplotlib import pyplot as plt
import seaborn as sns

params = {"SRAgent": {10: [3,4,5], 20:[3,4,5,6]}}
savedir = f"results/figs/kde"
file_prefix = ""

def get_paths(bm_size, agent, table_widths):
    paths = list(Path("results/pickles/travel_dists").glob(f"BM{bm_size}_*{agent}*a0.1*g0.9*greedy0.1*.pickle"))
    paths = [path for path in paths if int(re.search(r'table_width_(\d*)\D', path.name).group(1)) in table_widths]
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
def plot(bm_size, agent, table_widths):
    paths, pathr = get_paths(bm_size, agent, table_widths)
    plt.rcParams.update({'font.size': 16})
    cmap = plt.cm.turbo
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    paths = sorted(paths, key=lambda x: int(re.search(r'table_width_(\d+)', x.name).group(1)))
    for i, path in enumerate(paths):
        distr = distribution(path)
        table_width = int(re.search(r'table_width_(\d*)\D', path.name).group(1))
        sns.kdeplot(distr, label=f'tSiz={table_width}', log_scale=True, c=cmap(i/len(paths)))
        print("plotting", path.name)
    distr_r = distribution(pathr[0])
    sns.kdeplot(distr_r, color="gray",label='Random', linestyle="--", log_scale=True)
    print("plotting", pathr[0].name)

    # 凡例をプロットに追加
    plt.legend()
    ax.set_xlabel("Travel Distance")
    ax.set_ylabel("Density")
    ax.set_title(f"BM{bm_size} {agent}")
    path = f"{savedir}/{file_prefix}{bm_size}_{agent}.png"
    plt.savefig(f"{path}")
    print(f"{path} saved")

for agent_type in params.keys():
    for bm_size in params[agent_type].keys():
        table_widths = params[agent_type][bm_size]
        plot(bm_size, agent_type, table_widths)
