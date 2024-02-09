import numpy as np
from . import Agent

class RandomAgent(Agent):
    def __init__(self, env, *args, **kwargs) -> None:
        super().__init__(env)

    # stepはランダムに選択
    def step(self, observation, *args, **kwargs):
        actions = observation["next_actions"]
        return np.random.choice(actions)
    
    # updateは何もしない
    def update(self, *args, **kwargs):
        pass
