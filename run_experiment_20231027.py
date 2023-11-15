import datetime
import os
import pickle

import numpy as np
from agents import Agent, RandomAgent, QLearningAgent, SRAgent
from envs import BM, BM1, BM3, BM5, BM10, BM50, BM20
from analysis import Analysis
from multiprocessing import Pool, freeze_support
from utils import save_bm
import pyfbi

# parameters constant throughout experiment
n_agents = 1
alpha, gamma, epsilon = 0.1, 0.9, 0.1
policy = {"name": "e_greedy", "e": epsilon}
max_step = float("inf")
habituation_max_step = max_step
memorize_trajectory = True
# parameters
envs = [(BM10, 2)]
agents = [QLearningAgent]
do_habituation = [False]
table_widths = [3]
# other parameters
multiprocessing = False
save_pickle = False
pickle_preffix = "RoopTest"
save_result = False
savedir = "results/2023-10-20"

# do experiment for each agent and each environment
def do_experiment(agent_id:int, Env:BM, Agt:Agent, habituation:bool, n_episodes:int, table_width:int):
    env:BM = Env(max_step=max_step, habituation_max_step=habituation_max_step, memorize_trajectory=memorize_trajectory, goal_index=2)
    agent: Agent = Agt(env, table_width=table_width, alpha=alpha, gamma=gamma, policy=policy)
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
            if ep == 1:
                with open('b.txt', 'a') as f:
                    f.write(f"{agent.q_table} {action} {np.argmax(agent.q_table[0])} {observation['next_actions']}\n")
                print(env.step_count, agent.q_table, action, np.argmax(agent.q_table[0]), env.goal_index, observation["next_actions"])
            agent.update(observation)
            if observation["done"]:
                break
        print(f"        agent {agent_id} episode {ep} finished with {env.step_count} steps {Env.__name__} {Agt.__name__} {table_width} {alpha} {gamma} {policy['e']}")
    return agent, env

if __name__ == "__main__":
    os.makedirs(savedir, exist_ok=True)
    if multiprocessing:
        freeze_support()
    parallel_args = [(Env, n_episodes, Agent, habituation) for Env, n_episodes in envs for Agent in agents for habituation in do_habituation]
    for Env, n_episodes, Agt, habituation in parallel_args:
        exp_name = f"{Env.__name__}_{Agt.__name__}_{'habituation' if habituation else 'no_habituation'}_n{n_agents}_ne{n_episodes}_a{alpha}_g{gamma}_p{policy['name']}{policy['e']}"
        analysis = Analysis(exp_name, savedir)
        print(f"start {exp_name}")
        for table_width in table_widths:
            print(f"table_width: {table_width}")
            args = [(i, Env, Agt, habituation, n_episodes, table_width) for i in range(n_agents)]
            # do experiments for each params
            if multiprocessing:
                with Pool(n_agents) as p:
                    res = p.starmap(do_experiment, args)
                print("results generated")
            else:
                res = []
                for arg in args:
                    res.append(do_experiment(*arg))
                print("results generated")
            
            if save_pickle:
                print("saving pickle...")
                path_name = f"{pickle_preffix}_{exp_name}_table_width_{table_width}{'_trajOFF' if not memorize_trajectory else ''}"
                with open(f"results/pickles/{path_name}.pickle", "wb") as f:
                    pickle.dump(res, f)
            if save_result:
                print("appending to analysis...")
                analysis.append(envs, f"table_width_{table_width}")
        # analyze results and save
        if save_result:
            print("plotting conventionals group...")
            analysis.plot_conventionals_group()
    pyfbi.dump(f"stats/20231019/profile_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
