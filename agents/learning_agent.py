import numpy as np
import pickle
from datetime import datetime
from . import BaseAgent
from envs import BM
from utils import max_list

class LearningAgent(BaseAgent):
    def __init__(self, env:BM, agent_type:str, table_size:int=100, alpha:float=0.1, gamma:float=0.9, epsilon:float=0.1, policy={"name":"e_greedy", "e":0.1}):
        super().__init__(env)
        self.n_actions = env.n_actions
        self.table_size = table_size
        # observation_to_stateで使う値を計算しておく
        self.table_h = int(np.sqrt(table_size))
        self.table_w = self.table_h
        self.env_h = env.env_size[1]
        self.env_w = env.env_size[0]
        self.table_h_unit = self.env_h / self.table_h
        self.table_w_unit = self.env_w / self.table_w
        # agentの種類
        self.agent_type = agent_type
        # parameter
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy = policy

    def observation_to_state(self, observation):
        # observationのstateをtable_size分のgridで離散化する
        x, y = observation["state"]
        # stateは 6 | 7 | 8
        #         3 | 4 | 5
        #         0 | 1 | 2 のように区切られる。
        state = x // self.table_w_unit + self.table_w * (y // self.table_h_unit)
        return int(state)
    
    def save(self, prefix:str=datetime.now().strftime("%Y%m%d%H%M%S")):
        filename = f"models/{self.agent_type}_{prefix}_nActions{self.action_space.n}_tableSize{self.table_size}_alpha{self.alpha}_gamma{self.gamma}_epsilon{self.epsilon}"
        with open(f"{filename}.pickle", "wb") as f:
            pickle.dump(self, f)

    def load(self, path:str):
        with open(path, "rb") as f:
            obj =  pickle.load(f)
        self = obj

    def step(self, observation):
        state = self.observation_to_state(observation)
        actions = observation["next_actions"]
        # e-greedy
        if self.policy["name"] == "e_greedy":
            if np.random.random() < self.policy["e"]:
                action = np.random.choice(actions)
            else:
                # 現在のstateにおけるq値を取得。[q(s, a1), q(s, a2), ...]
                q_values = self.q_values_on_s(state)
                # actionsの中にある、最大のq値を持つactionを取得。最大値が複数ある場合はランダムに選択。
                max_action_list = max_list([(q_values[a], a) for a in actions], key=lambda x:x[0])
                if len(max_action_list) == 1: # 最大値が1つだけなら
                    action = max_action_list[0][1]
                else: # 最大値が複数あるなら
                    action = np.random.choice([a for _, a in max_action_list])
        # boltzmann
        elif self.policy["name"] == "boltzmann":
            # 現在のstateにおけるq値を取得。[q(s, a1), q(s, a2), ...]。（actionsに存在する可能なアクションに限っている）
            q_values = np.array([q for a, q in enumerate(self.q_values_on_s(state)) if a in actions])
            # softmax法で確率分布を計算
            q_values = np.exp(q_values / self.policy["t"])
            q_values = q_values / np.sum(q_values)
            action = np.random.choice(actions, p=q_values)
        else:
            raise ValueError(f"policy name {self.policy['name']} is not defined")
        self.last_action = action
        self.last_state = state
        return action

    def q_values_on_s(self, state):
        pass

    def update(self, observation):
        state = self.observation_to_state(observation)
        reward = observation["reward"]
        self.update_table(state, reward)

    def update_table(self, state, reward):
        pass