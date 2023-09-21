from gymnasium import Env
import pickle

class BaseAgent:
    def __init__(self, env:Env) -> None:
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.observation_sup = env.observation_sup
        self.observation_inf = env.observation_inf

    def save(self, path:str):
        with open(path, "wb") as f:
            pickle.dump(self, f)