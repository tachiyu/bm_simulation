from pathlib import Path
import pickle
from envs import BM

from analysis import Analysis

if __name__ == "__main__":
    paths = Path("results/pickle").glob("*.pickle")
    a = Analysis()
    for path in paths:
        print(f"extracting {path}")
        if "trajOFF" in path.name:
            a.exstract_travel_distance_trajOff(path, savedir="results/travel_dist/pickle")
        else:
            with open(path, "rb") as f:
                bms:list[BM] = pickle.load(f)
            tds = [[len(tr) for tr in bm.trajectory_list] for bm in bms]
            with open(f"results/travel_dist/pickle/{path.name}", "wb") as f:
                pickle.dump(tds, f)