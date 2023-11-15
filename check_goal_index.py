from pathlib import Path
import pickle

from envs import BM


def main():
    paths = list(Path("results/pickle").glob(f"*trajOFF*.pickle"))
    for path in paths:
        with open(path, "rb") as f:
            bms = pickle.load(f)
        goal_index_cnt = {f"{x}" : 0 for x in range(12)}
        for bm in bms:
            if "withAgent" in path.name:
                _, bm = bm
            goal_index_cnt[f"{bm.goal_index}"] += 1
        print(path.name, [f"{x}: {goal_index_cnt[f'{x}']}" for x in range(12) if goal_index_cnt[f"{x}"] != 0])

main()