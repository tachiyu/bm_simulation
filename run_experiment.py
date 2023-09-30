import os
from agents import RandomAgent, QLearningAgent, SRAgent
from envs import BM, BM1, BM3
from experiment import Experiment
from multiprocessing import Pool, freeze_support
import pyfbi

n_agents = 50
multiprocessing = True
envs = [(BM1, 3*6), (BM3, 3*12)]
agents = [SRAgent, QLearningAgent, RandomAgent]
do_habituation = [False, True]

table_size = 100
alpha = 0.1
gamma = 0.9
# policy = {"name": "boltzmann", "t": 0.01}
policy = {"name": "e_greedy", "e": 0.1}
max_step = 100000
habituation_max_step = 50000

savedir = "results/2023-09-27"
os.makedirs(savedir, exist_ok=True)

def multi_func(i, Env: BM, Agent, habituation, n_episodes, table_size, alpha, gamma, policy):
    env = Env(max_step=max_step, habituation_max_step=habituation_max_step)
    agent = Agent(env, table_size=table_size, alpha=alpha, gamma=gamma, policy=policy)
    # print(f"    agent {i} start. goal_index: {env.goal_index}")
    if habituation:
        observation = env.reset(episode=-1)
        while True:
            action = agent.step(observation)
            observation = env.step(action)
            agent.update(observation)
            if observation["done"]:
                break
        print(f"        agent {i} habituation finished with {env.step_count} steps")
    for ep in range(n_episodes):
        observation = env.reset(episode=ep)
        while True:
            action = agent.step(observation)
            observation = env.step(action)
            agent.update(observation)
            if observation["done"]:
                break
            if ep == 0 and not multiprocessing:
                env.render()
        print(f"        agent {i} episode {ep} finished with {env.step_count} steps")
        # agent.show_qtable_max()
        # agent.show_qtable()
        # print(max(agent.rewards))
    return env

if __name__ == "__main__":
        if multiprocessing:
            freeze_support()
        for Env, n_episodes in envs:
            for Agent in agents:
                for habituation in do_habituation:
                    exp_name = f"{Env.__name__}_{Agent.__name__}_{'habituation' if habituation else 'no_habituation'}_n{n_agents}_ne{n_episodes}_tSiz{table_size}_a{alpha}_g{gamma}_p{policy['name']}"
                    experiment = Experiment(exp_name, savedir)
                    print(f"start {exp_name}")
                    args = [(i, Env, Agent, habituation, n_episodes, table_size, alpha, gamma, policy) for i in range(n_agents)]
                    if multiprocessing:
                        with Pool(n_agents) as p:
                            results = p.starmap(multi_func, args)
                        print("results generated")
                    else:
                        results = []
                        for arg in args:
                            results.append(multi_func(*arg))
                    print("results generated")
                    for env in results:
                        experiment.append(env)
                    
                    print("plotting conventionals...")
                    experiment.plot_conventionals()
                    print("plotting strategy...")
                    experiment.plot_strategy()