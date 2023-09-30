from gymnasium import Env
import pickle

class BaseAgent:
    def __init__(self, env:Env) -> None:
        pass

    def step(self, observation):
        pass

    def update(self, observation):
        pass