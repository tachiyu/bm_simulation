from agents.base import BaseAgent
from gymnasium import Env


class RandomAgent(BaseAgent):
    def __init__(self, env:Env) -> None:
        super().__init__(env)

    def step(self, state, reward, done, info):
        return self.action_space.sample()
