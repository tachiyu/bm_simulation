import pickle
from envs import BM


path = r"C:\Users\bdr\Desktop\Projects\bm_simulation\results\pickles\RoopTest_BM3_QLearningAgent_no_habituation_n10_ne2_a0.1_g0.9_pe_greedy0.1_table_width_1.pickle"
with open(path, 'rb') as f:
    data: list[BM] = pickle.load(f)

data[2].replay_trajectory(1, 600000)