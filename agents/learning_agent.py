import numpy as np
import pickle
from datetime import datetime
from . import Agent
from envs import BM

# すべての学習エージェントの基底クラス。
class LearningAgent(Agent):
    def __init__(self, env:BM, agent_type:str, table_width:int, alpha:float, gamma:float, policy:dict):
        self.n_actions = env.n_actions # actionの数
        self.n_states = table_width ** 2 # stateの数

        # __observation_to_state関数で使う値を計算しておく (計算量をおさえるため)
        self.table_h = table_width # tableの縦のサイズ
        self.table_w = table_width # tableの横のサイズ
        self.env_h = env.env_size[1] # 環境の縦のサイズ
        self.env_w = env.env_size[0] # 環境の横のサイズ
        self.table_h_unit = self.env_h / self.table_h # tableの1マスあたりの環境の縦のサイズ
        self.table_w_unit = self.env_w / self.table_w  # tableの1マスあたりの環境の横のサイズ

        self.agent_type = agent_type # エージェントの種類
        self.alpha = alpha # 学習率
        self.gamma = gamma # 割引率
        self.policy = policy # 方策

    # stepを実装 (observationを受け取り、actionを返す)
    def step(self, observation):
        state = self._observation_to_state(observation) # 現在の脳内におけるstateを取得
        actions = observation["next_actions"] # 可能なactionのリスト

        # policyに従ってactionを選択
        if self.policy["name"] == "e_greedy": # e-greedy
            if np.random.random() < self.policy["e"]:
                action = np.random.choice(actions)
            else:
                # 現在のstateにおけるq値の配列を取得。[q(s, a1), q(s, a2), ...]
                q_values = self._q_values_on_s(state)
                # actionsの中にある、最大のq値を持つactionのリストを取得。
                valid_q_a_pairs = [(q_values[a], a) for a in actions] # 可能なactionのq値のペアのリスト
                max_actions = LearningAgent._get_max_actions(valid_q_a_pairs)
                if len(max_actions) == 1: # 最大値が1つだけなら
                    action = max_actions[0]
                else: # 最大値が複数あるならランダムに選ぶ
                    action = np.random.choice(max_actions)
        elif self.policy["name"] == "boltzmann": # boltzmann
            # 現在のstateにおけるq値を取得。[q(s, a1), q(s, a2), ...]
            q_values = self._q_values_on_s(state)
            valid_q_values = [q_values[a] for a in actions] # 可能なactionのq値のリスト
            # softmax法でactionの確率分布を計算
            probs = np.exp(valid_q_values / self.policy["t"])
            probs = valid_q_values / np.sum(valid_q_values)
            # 確率分布に従ってactionを選択
            action = np.random.choice(actions, p=probs)
        else:
            raise ValueError(f"policy name {self.policy['name']} is not defined")
        
        self.last_action = action
        self.last_state = state
        return action
    
    # updateを実装 (observationを受け取り、学習を行う)
    def update(self, observation):
        state = self._observation_to_state(observation)
        reward = observation["reward"]
        self._update_table(state, reward)

    # あるstateにおける、q値のリストを返す (継承先で実装)。
    def _q_values_on_s(self, state):
        pass

    # あるstateとrewardにおいて、テーブルを更新する (継承先で実装)。
    def _update_table(self, state, reward):
        pass

    # observationをエージェントの脳内におけるstateに変換する
    def _observation_to_state(self, observation):
        # stateは 6 | 7 | 8
        #         3 | 4 | 5
        #         0 | 1 | 2 のようにグリッドで区切られ離散化される。

        # observationのstateをn_states分のgridで離散化する
        x, y = observation["state"]
        state = (x+self.env_w/2)//self.table_w_unit + self.table_w*((y+self.env_h/2)//self.table_h_unit)
        return int(state)

    # listの中から最大のq値を持つactionを返す。リストとして返すことに注意。
    @staticmethod
    def _get_max_actions(l:list):
        max_value = max(l, key=lambda x:x[0])[0]
        return [a for v, a in l if v == max_value]
    
    # 環境が変わったときに呼び出される (メタ学習用)
    def reset_env(self, env:BM, *args, **kwargs):
        self.env_h = env.env_size[1]
        self.env_w = env.env_size[0]
        self.table_h_unit = self.env_h / self.table_h
        self.table_w_unit = self.env_w / self.table_w