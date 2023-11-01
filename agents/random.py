import numpy as np
from . import Agent

class RandomAgent(Agent):
    def __init__(self, env, *args, **kwargs) -> None:
        super().__init__(env)

    def step(self, observation, *args, **kwargs):
        actions = observation["next_actions"]
        return np.random.choice(actions)
    
    def update(self, *args, **kwargs):
        pass
