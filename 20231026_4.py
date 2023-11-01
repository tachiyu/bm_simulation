from pathlib import Path

from analysis import Analysis

if __name__ == "__main__":
    paths = Path("results/pickles").glob("*trajOFF.pickle")
    a = Analysis()
    for path in paths:
        print(f"extracting {path}")
        a.exstract_travel_distance_trajOff(path)