import numpy as np
import matplotlib.pyplot as plt

from . import LearningAgent

# SR学習エージェント
class SRAgent(LearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, "SR", **kwargs) # LearningAgentの__init__を実行

        self.sr_table = np.zeros((self.n_states, self.n_states, self.n_actions)) # SRテーブルの初期化
        self.rewards = np.zeros(self.n_states) # rewardテーブルの初期化
    
    # あるstateにおける、q値のリストを返す (SRテーブルとrewardテーブルの掛け算)
    def _q_values_on_s(self, state):
        return np.dot(self.rewards, self.sr_table[state])

    # あるstateとrewardにおいて、SRテーブルとrewardテーブルを更新する。
    def _update_table(self, state, reward):
        # 報酬テーブルの更新
        # r(s) = r
        self.rewards[state] = reward
        # SRテーブルの更新
        # m(s, s', a) = (1 - alpha) * m(s, s', a) + alpha * (I(s==s') + gamma * max_a(m(s, s', a)))
        # すべてのs'について計算。
        # 計算量を減らすため、更新式と少し違うが、結果は同じ。
        new_sr_row = self.sr_table[self.last_state, :, self.last_action]
        new_sr_row *= (1 - self.alpha)
        alpha_term = self.sr_table[state].max(axis=1)
        alpha_term *= self.alpha * self.gamma
        alpha_term[state] += self.alpha 
        new_sr_row += alpha_term
        self.sr_table[self.last_state, :, self.last_action] = new_sr_row

    # Qテーブル (SR x R) の可視化
    def show_qtable(self):
        # 1つのstateを9マスで割り、色でq値を表現する
        table = np.zeros((self.table_h * 3, self.table_w * 3))
        q_table = np.dot(self.rewards, self.sr_table.transpose([1, 0, 2]))
        action_to_position = [(1,2),(2,2),(2,1),(2,0),(1,0),(0,0),(0,1),(0,2),(1,1)]
        for i in range(self.table_h):
            for j in range(self.table_w):
                for a in range(self.n_actions):
                    table[i*3 + action_to_position[a][0], j*3 + action_to_position[a][1]] = q_table[i*self.table_h + j, a]
        plt.imshow(table, origin="lower")
        # max color is 1e-5
        plt.clim(0, 1)
        plt.colorbar()
        plt.show()

    # Qテーブル (SR x R) の可視化、最大値のみ    
    def show_qtable_max(self):
        # 1つのstateを9マスで割り、色でq値を表現する
        q_table = np.dot(self.rewards, self.sr_table.transpose([1, 0, 2]))
        table = np.zeros((self.table_h * 3, self.table_w * 3))
        action_to_position = [(1,2),(2,2),(2,1),(2,0),(1,0),(0,0),(0,1),(0,2),(1,1)]
        for i in range(self.table_h):
            for j in range(self.table_w):
                max_a = max([(q_table[i*self.table_h + j, a], a) for a in range(self.n_actions)], key=lambda x:x[0])[1]
                table[i*3+1, j*3+1] = 1
                table[i*3 + action_to_position[max_a][0], j*3 + action_to_position[max_a][1]] = 0.5
        plt.imshow(table, origin="lower")
        plt.colorbar()
        plt.show()