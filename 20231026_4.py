from pathlib import Path

from analysis import Analysis

if __name__ == "__main__":
    paths = Path("results/2023-10-20/pickle").glob("*.pickle")
    a = Analysis()
    for path in paths:
        print(f"extracting {path}")
        a.exstract_travel_distance_trajOff(path, savedir="results/2023-10-20/travel_distance")