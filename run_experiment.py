import os
from agents import Agent, RandomAgent, QLearningAgent, SRAgent
from envs import BM, BM1, BM3
from analysis import Analysis
from multiprocessing import Pool, freeze_support
import pyfbi

# parameters constant throughout experiment
n_agents = 1
alpha = 0.1
gamma = 0.9
policy = {"name": "e_greedy", "e": 0.1}
max_step = 100000
habituation_max_step = 100000
table_width = 10
# parameters
envs = [(BM1, 3*6), (BM3, 3*12)]
agents = [SRAgent, QLearningAgent]
do_habituation = [True]
# other parameters
multiprocessing = False
render = True
save_result = False
savedir = "results/2023-10-16"

# do experiment for each agent and each environment
def do_experiment(agent_id:int, Env:BM, Agt:Agent, habituation:bool, n_episodes:int, table_width:int, alpha:float, gamma:float, policy:dict):
    env = Env(max_step=max_step, habituation_max_step=habituation_max_step)
    agent = Agt(env, table_width=table_width, alpha=alpha, gamma=gamma, policy=policy)
    # first, do habituation
    if habituation:
        observation = env.reset(episode=-1)
        while True:
            action = agent.step(observation)
            observation = env.step(action)
            agent.update(observation)
            if observation["done"]:
                break
        print(f"        agent {agent_id} habituation finished with {env.step_count} steps")
    # second, do training
    for ep in range(n_episodes):
        observation = env.reset(episode=ep)
        while True:
            action = agent.step(observation)
            observation = env.step(action)
            agent.update(observation)
            if observation["done"]:
                break
        print(f"        agent {agent_id} episode {ep} finished with {env.step_count} steps")
    return env

if __name__ == "__main__":
    os.makedirs(savedir, exist_ok=True)
    if multiprocessing:
        freeze_support()

    parallel_args = [(Env, n_episodes, Agent, habituation) for Env, n_episodes in envs for Agent in agents for habituation in do_habituation]
    for Env, n_episodes, Agt, habituation in parallel_args:
        exp_name = f"{Env.__name__}_{Agt.__name__}_{'habituation' if habituation else 'no_habituation'}_n{n_agents}_ne{n_episodes}_tSiz{table_size}_a{alpha}_g{gamma}_p{policy['name']}"
        analysis = Analysis(exp_name, savedir)
        print(f"start {exp_name}")
        args = [(i, Env, Agt, habituation, n_episodes, table_size, alpha, gamma, policy) for i in range(n_agents)]
        # do experiments for each params
        if multiprocessing:
            with Pool(n_agents) as p:
                results = p.starmap(do_experiment, args)
            print("results generated")
        else:
            results = []
            for arg in args:
                results.append(do_experiment(*arg))
            print("results generated")
        # analyze results and save
        if save_result:
            for env in results:
                analysis.append(env)
            print("plotting conventionals...")
            analysis.plot_conventionals()
            print("plotting strategy...")
            analysis.plot_strategy()  