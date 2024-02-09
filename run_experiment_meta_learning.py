import os
import pickle

import numpy as np

from agents import Agent, RandomAgent, QLearningAgent, SRAgent
from envs import BM, BM1, BM3, BM5, BM10, BM50, BM20
from multiprocessing import Pool, freeze_support

# 下のパラメータで実験を行い、その結果をpickleで保存する。
# 実験の結果は、savedirにBMオブジェクトが、savedir2に総移動距離が保存される。
# 複数のエージェントの結果が保存されるため、pickleの中身はリストになっている。

n_agents = 2 # エージェント数
alpha, gamma, epsilon = 0.1, 0.9, 0.1 # 学習率、割引率、ε-greedyのε
policy = {"name": "e_greedy", "e": epsilon} # 方策
max_step = float("inf") # 最大ステップ数
habituation_max_step = max_step # habituationの最大ステップ数
memorize_trajectory = False # 軌跡をBMオブジェクトに記録するか
# おこなう実験の条件   
conditions = [
    {"table_width": 6, "agent": SRAgent, "envs": [(BM1, 3*12), (BM10, 3*12)]},
    {"table_width": 4, "agent": QLearningAgent, "envs": [(BM1, 3*12), (BM10, 3*12)]},
    {"table_width": 19, "agent": SRAgent, "envs": [(BM10, 3*12), (BM1, 3*12)]},
    {"table_width": 6, "agent": QLearningAgent, "envs": [(BM10, 3*12), (BM1, 3*12)]},
    {"table_width": 6, "agent": QLearningAgent, "envs": [(BM1, 3*12), (BM1, 3*12)]},
    {"table_width": 19, "agent": SRAgent, "envs": [(BM1, 3*12), (BM1, 3*12)]},
    {"table_width": 4, "agent": QLearningAgent, "envs": [(BM1, 3*12), (BM1, 3*12)]},
    {"table_width": 6, "agent": SRAgent, "envs": [(BM1, 3*12), (BM1, 3*12)]},
    {"table_width": 6, "agent": QLearningAgent, "envs": [(BM10, 3*12), (BM10, 3*12)]},
    {"table_width": 19, "agent": SRAgent, "envs": [(BM10, 3*12), (BM10, 3*12)]},
    {"table_width": 4, "agent": QLearningAgent, "envs": [(BM10, 3*12), (BM10, 3*12)]},
    {"table_width": 6, "agent": SRAgent, "envs": [(BM10, 3*12), (BM10, 3*12)]},
]
do_habituation = [[False, False]] # habituationを行うかどうかのリスト (課題につき一つずつ)
render = False # 実験の様子実行中にを描画するか (multi-processingと同時には使えない)
multiprocessing = True # multi-processingを使うか
save_pickle = True # pickleを保存するか
pickle_preffix = "Meta_" # pickleの接頭辞
savedir = "results/pickles/bms" # BMオブジェクトを保存するディレクトリ
savedir2 = "results/pickles/travel_dists" # 総移動距離を保存するディレクトリ

# １つのエージェントがある環境で実験を行う。この関数をmulti-processingして使う。
def do_experiment(agent_id:int, Envs:list[BM], Agt:Agent, habituations:list[bool], n_episodes:list[int], table_width:int):
    goal_index = np.random.randint(12)
    env:BM = Envs[0](max_step=max_step, habituation_max_step=habituation_max_step, memorize_trajectory=memorize_trajectory)
    agent: Agent = Agt(env, table_width=table_width, alpha=alpha, gamma=gamma, policy=policy)
    def do_episode(observation, agent, env, render=False):
        while True:
            action = agent.step(observation)
            observation = env.step(action)
            agent.update(observation)
            if render:
                env.render()
            if observation["done"]:
                break

    # first learining
    # first, do habituation
    if habituations[0]:
        observation = env.reset(episode=-1)
        do_episode(observation, agent, env, render)
        print(f"        agent {agent_id} habituation finished with {env.step_count} steps")
    # second, do training
    for ep in range(n_episodes[0]):
        observation = env.reset(episode=ep)
        do_episode(observation, agent, env, render)
        print(f"        agent {agent_id} episode {ep} finished with {env.step_count} steps {Envs[0].__name__} {Agt.__name__} {table_width} {alpha} {gamma} {policy['e']}  goal={goal_index}")
    
    # second learning
    # first, do habituation
    goal_index2 = np.random.choice([x for x in range(12) if x != goal_index])
    env2:BM = Envs[1](max_step=max_step, habituation_max_step=habituation_max_step, memorize_trajectory=memorize_trajectory, goal_index=goal_index2)
    agent.reset_env(env2)
    if habituations[1]:
        observation = env2.reset(episode=-1)
        do_episode(observation, agent, env2, render)
        print(f"        agent {agent_id} habituation finished with {env2.step_count} steps")
    # second, do training
    for ep in range(n_episodes[1]):
        observation = env2.reset(episode=ep)
        do_episode(observation, agent, env2, render)
        print(f"        agent {agent_id} episode {ep} finished with {env2.step_count} steps {Envs[1].__name__} {Agt.__name__} {table_width} {alpha} {gamma} {policy['e']}  goal={goal_index2}")
    
    return env, env2

if __name__ == "__main__":
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir2, exist_ok=True)    
    if multiprocessing:
        freeze_support()
    parallel_args = [(c['envs'], c['agent'], c['table_width'], habituation) for c in conditions for habituation in do_habituation]
    for envs, Agt, table_width, habituation in parallel_args:
        n_episodes = [e[1] for e in envs]
        Envs = [e[0] for e in envs]
        exp_name = f"Meta_{Envs[0].__name__}to{Envs[1].__name__}_{Agt.__name__}_{'habituation' if habituation else 'no_habituation'}_n{n_agents}_ne{n_episodes[0]}to{n_episodes[1]}_a{alpha}_g{gamma}_p{policy['name']}{policy['e']}"
        print(f"start {exp_name}")
        print(f"table_width: {table_width}")
        args = [(i, Envs, Agt, habituation, n_episodes, table_width) for i in range(n_agents)]
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
            path_name = f"{pickle_preffix}{exp_name}_table_width_{table_width}{'_trajOFF' if not memorize_trajectory else ''}"
            with open(f"{savedir}/{path_name}.pickle", "wb") as f:
                pickle.dump(res, f)
            with open(f"{savedir2}/{path_name}.pickle", "wb") as f:
                tds = [(bm.step_count_list, bm2.step_count_list) for bm, bm2 in res]
                pickle.dump(tds, f)
