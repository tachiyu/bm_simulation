import datetime
import os
import pickle
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

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
memorize_trajectory = False
# parameters
envs = [(BM20, 3*12)]
agents = [QLearningAgent]
do_habituation = [False]
table_widths = [2,3,1]
# other parameters
multiprocessing = False
save_pickle = True
pickle_preffix = ""
save_result = False
savedir = "to_progress_report"
os.makedirs(savedir, exist_ok=True)

# do experiment for each agent and each environment
def do_experiment(agent_id:int, Env:BM, Agt:Agent, habituation:bool, n_episodes:int, table_width:int):
    env:BM = Env(max_step=max_step, habituation_max_step=habituation_max_step, memorize_trajectory=memorize_trajectory)
    agent: Agent = Agt(env, table_width=table_width, alpha=alpha, gamma=gamma, policy=policy)
    # second, do training
    fig = plt.figure()
    for ep in range(n_episodes):
        observation = env.reset(episode=ep)
        frames = []
        c = 0
        while True:
            c += 1
            action = agent.step(observation)
            observation = env.step(action)
            agent.update(observation)
            if observation["done"]:
                break
            if c < 1000 and ep == 0 or ep == 17 or ep == 35:
                frames.append(env.render(pause=False))
        if ep == 0 or ep == 17 or ep == 35:
            anim = ArtistAnimation(fig, frames, interval=50, repeat_delay=1000, blit=True)
            anim.save(f"{savedir}/Anim_episode{ep}_{Env.__name__}_{Agt.__name__}_{'habituation' if habituation else 'no_habituation'}_n{n_agents}_ne{n_episodes}_a{alpha}_g{gamma}_p{policy['name']}{policy['e']}_table_width_{table_width}.mp4", writer="ffmpeg")
        print(f"        agent {agent_id} episode {ep} finished with {env.step_count} steps {Env.__name__} {Agt.__name__} {table_width} {alpha} {gamma} {policy['e']}")
    return agent, env

do_experiment(0, BM1, SRAgent, False, 36, 8)
