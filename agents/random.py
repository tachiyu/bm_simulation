import numpy as np
from gymnasium import Env

from . import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, env:Env, *args, **kwargs) -> None:
        super().__init__(env)

    def step(self, observation, *args, **kwargs):
        actions = observation["next_actions"]
        return np.random.choice(actions)
    
    def update(self, *args, **kwargs):
        pass
