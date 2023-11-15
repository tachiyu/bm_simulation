from pathlib import Path
import pickle
from envs import BM

from analysis import Analysis

if __name__ == "__main__":
    path = Path(r"C:\Users\bdr\Desktop\Projects\bm_simulation\results\pickle\BM20_QLearningAgent_no_habituation_n30_ne36_a0.1_g0.9_pe_greedy0.1_table_width_1_trajOFF_withAgent.pickle")
    a = Analysis()
    print(f"extracting {path}")
    if "trajOFF" in path.name:
        a.exstract_travel_distance_trajOff(path, savedir="results/travel_dist/pickle")
