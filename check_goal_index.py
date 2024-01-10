import datetime
from pathlib import Path
import pickle

from envs import BM


def main():
    paths = list(Path("results/pickle").glob(f"*trajOFF*.pickle"))
    with open(f"goal_index_cnt_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt", "w") as of:
        for path in paths:
            with open(path, "rb") as f:
                bms = pickle.load(f)
            goal_index_cnt = {x : 0 for x in range(12)}
            for bm in bms:
                if "withAgent" in path.name:
                    _, bm = bm
                goal_index_cnt[bm.goal_index] += 1
            of.write(f"{path.name}  {[(x, goal_index_cnt[x]) for x in range(12) if goal_index_cnt[x] != 0]} \n")

main()