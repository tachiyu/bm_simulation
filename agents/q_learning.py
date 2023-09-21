import numpy as np
from gymnasium import Env
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

from . import BaseAgent
from utils import max_list

class QLearningAgent(BaseAgent):
    def __init__(self, env:Env, table_size:int=1000, alpha:float=0.1, gamma:float=0.9, policy={"name":"e_greedy", "e":0.1}):
        super().__init__(env)
        self.q_table_size = table_size
        self.n_actions = self.action_space.n
        # self.q_table = np.random.uniform(low=0, high=1e-9, size=(table_size, self.n_actions))
        self.q_table = np.zeros((table_size, self.n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy

    
    def save(self, prefix:str=datetime.now().strftime("%Y%m%d%H%M%S")):
        filename = f"models/qAgent_{prefix}_nActions{self.action_space.n}_qTableSize{self.q_table_size}_alpha{self.alpha}_gamma{self.gamma}_epsilon{self.epsilon}"
        with open(f"{filename}.pickle", "wb") as f:
            pickle.dump(self, f)

    def observation_to_state(self, observation):
        # observationのstateをq_table_size分のgridで離散化する
        observed_state = observation["state"]
        env_high = self.observation_sup[0]
        q_table_size_root = int(np.sqrt(self.q_table_size))
        # stateは 6 | 7 | 8
        #         3 | 4 | 5
        #         0 | 1 | 2 のように区切られる。
        state = observed_state[0] // (env_high / q_table_size_root) \
                + q_table_size_root * (observed_state[1] // (env_high / q_table_size_root))
        
        return int(state)
    
    def step(self, observation):
        state = self.observation_to_state(observation)
        actions = observation["next_actions"]
        # e-greedy
        if self.policy["name"] == "e_greedy":
            if np.random.random() < self.policy["e"]:
                action = np.random.choice(actions)
            else:
                max_action_list = max_list([(self.q_table[state, a], a) for a in actions], key=lambda x:x[0])
                if len(max_action_list) == 1:
                    action = max_action_list[0][1]
                else:
                    action = np.random.choice([a for _, a in max_action_list])
        # boltzmann
        elif self.policy["name"] == "boltzmann":
            # softmax
            q_values = np.array([self.q_table[state, a] for a in actions])
            q_values = np.exp(q_values / self.policy["t"])
            q_values = q_values / np.sum(q_values)
            action = np.random.choice(actions, p=q_values)
        else:
            raise ValueError(f"policy name {self.policy['name']} is not defined")
        self.last_action = action
        self.last_state = state
        return action
    
    def update(self, observation):
        state = self.observation_to_state(observation)
        reward = observation["reward"]
        self.q_table[self.last_state, self.last_action] = (1 - self.alpha) * self.q_table[self.last_state, self.last_action] + self.alpha * (reward + self.gamma * np.max(self.q_table[state]))

    def show_qtable(self):
        # q_tableの可視化
        # 1つのstateを9マスで割り、色でq値を表現する
        q_table_size_root = int(np.sqrt(self.q_table_size))
        table = np.zeros((q_table_size_root * 3, q_table_size_root * 3))
        action_to_position = [(1,2),(2,2),(2,1),(2,0),(1,0),(0,0),(0,1),(0,2),(1,1)]
        for i in range(q_table_size_root):
            for j in range(q_table_size_root):
                for a in range(self.n_actions):
                    table[i*3 + action_to_position[a][0], j*3 + action_to_position[a][1]] = self.q_table[i*q_table_size_root + j, a]
        plt.imshow(table, origin="lower")
        # max color is 1e-5
        # plt.clim(0, 1e-9)
        plt.colorbar()
        plt.show()

    def show_qtable_max(self):
        # q_tableの可視化. maxのみ
        # 1つのstateを9マスで割り、色でq値を表現する
        q_table_size_root = int(np.sqrt(self.q_table_size))
        table = np.zeros((q_table_size_root * 3, q_table_size_root * 3))
        action_to_position = [(1,2),(2,2),(2,1),(2,0),(1,0),(0,0),(0,1),(0,2),(1,1)]
        for i in range(q_table_size_root):
            for j in range(q_table_size_root):
                max_a = max([(self.q_table[i*q_table_size_root + j, a], a) for a in range(self.n_actions)], key=lambda x:x[0])[1]
                table[i*3+1, j*3+1] = 1
                table[i*3 + action_to_position[max_a][0], j*3 + action_to_position[max_a][1]] = 0.5
        plt.imshow(table, origin="lower")
        plt.colorbar()
        plt.show()

