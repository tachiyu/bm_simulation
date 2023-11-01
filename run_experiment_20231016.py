import datetime
import os
from agents import Agent, RandomAgent, QLearningAgent, SRAgent
from envs import BM, BM1, BM3, BM5, BM10, BM50, BM20
from analysis import Analysis
from multiprocessing import Pool, freeze_support
from utils import save_bm
import pyfbi

# parameters constant throughout experiment
n_agents = 30
hparams = [(0.1, 0.5, 0.1), (0.5, 0.9, 0.1), (0.1, 0.9, 0.5)]
policy = {"name": "e_greedy", "e": 0.1}
max_step = float("inf")
habituation_max_step = max_step
memorize_trajectory = False
# parameters
envs = [(BM3, 3*12), (BM5, 3*12), (BM10, 3*12)]
agents = [QLearningAgent, SRAgent]
do_habituation = [False]
table_widths = [5, 10, 30]
# other parameters
multiprocessing = True
save_pickle = True
save_result = False
savedir = "results/2023-10-20"

# do experiment for each agent and each environment
def do_experiment(agent_id:int, Env:BM, Agt:Agent, habituation:bool, n_episodes:int, table_width:int, alpha:float, gamma:float, policy:dict):
    env:BM = Env(max_step=max_step, habituation_max_step=habituation_max_step, memorize_trajectory=memorize_trajectory)
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
            agent.update(observation)
            if observation["done"]:
                break
        print(f"        agent {agent_id} episode {ep} finished with {env.step_count} steps {Env.__name__} {Agt.__name__} {table_width} {alpha} {gamma} {policy['e']}")
    return env

if __name__ == "__main__":
    os.makedirs(savedir, exist_ok=True)
    if multiprocessing:
        freeze_support()
    with pyfbi.watch(global_watch=True):
        parallel_args = [(Env, n_episodes, Agent, habituation) for Env, n_episodes in envs for Agent in agents for habituation in do_habituation]
        for Env, n_episodes, Agt, habituation in parallel_args:
            for alpha, gamma, epsilon in hparams:
                policy["e"] = epsilon
                exp_name = f"{Env.__name__}_{Agt.__name__}_{'habituation' if habituation else 'no_habituation'}_n{n_agents}_ne{n_episodes}_a{alpha}_g{gamma}_p{policy['name']}{policy['e']}"
                analysis = Analysis(exp_name, savedir)
                print(f"start {exp_name}")
                for table_width in table_widths:
                    print(f"table_width: {table_width}")
                    args = [(i, Env, Agt, habituation, n_episodes, table_width, alpha, gamma, policy) for i in range(n_agents)]
                    # do experiments for each params
                    if multiprocessing:
                        with Pool(n_agents) as p:
                            envs = p.starmap(do_experiment, args)
                        print("results generated")
                    else:
                        envs = []
                        for arg in args:
                            envs.append(do_experiment(*arg))
                        print("results generated")
                    exp_name_2 = f"{exp_name}_table_width_{table_width}{'_trajOFF' if not memorize_trajectory else ''}"
                    if save_pickle:
                        save_bm(envs, exp_name_2)
                    if save_result:
                        analysis.append(envs, f"table_width_{table_width}")
                # analyze results and save
                if save_result:
                    print("plotting conventionals group...")
                    analysis.plot_conventionals_group()
    pyfbi.show()
    pyfbi.dump(f"stats/20231019/profile_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
