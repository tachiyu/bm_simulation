import os
import pickle

def get_max_actions(l:list):
    max_value = max(l, key=lambda x:x[0])[0]
    return [a for v, a in l if v == max_value]

def save_bm(bm, name):
    path = f"results/pickles/{name}.pickle"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm, f)
