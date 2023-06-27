from gymnasium import spaces, Env

class BaseAgent:
    def __init__(self, env:Env) -> None:
        self.action_space = env.action_space
        self.observation_space = env.observation_space