from agents.random import RandomAgent
from envs.BM1 import BM1

env = BM1(goal_index=2)
env.reset()

agent = RandomAgent(env)

for i in range(100):
    print(env.state)
    action = agent.step(env.state, 0, False, set())
    env.step(action)
    env.render()
    if env.done:
        env.reset()