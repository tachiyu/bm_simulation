import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from numba import jit

from . import LearningAgent

class SRAgent(LearningAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, "SR", **kwargs)
        self.sr_table = np.zeros((self.table_size, self.table_size, self.n_actions))
        self.rewards = np.zeros(self.table_size)
    
    def q_values_on_s(self, state):
        return np.dot(self.rewards, self.sr_table[state])
    
    # def update_table(self, state, reward):
    #     # update reward table
    #     self.rewards[state] = reward
    #     # update sr table (usign cupy)
    #     new_sr_row = cp.array(self.sr_table[self.last_state, :, self.last_action])
    #     new_sr_row *= (1 - self.alpha)
    #     alpha_term = cp.array(self.sr_table[state]).max(axis=1)
    #     alpha_term *= self.alpha * self.gamma
    #     alpha_term[state] += self.alpha # this is equivalent for hot vector addition but faster
    #     new_sr_row += alpha_term
    #     self.sr_table[self.last_state, :, self.last_action] = new_sr_row.get()

    # def update_table(self, state, reward):
    #     # update reward table
    #     self.rewards[state] = reward
    #     # update sr table (usign cupy)
    #     hot = np.zeros(self.table_size)
    #     hot[state] = 1
    #     self.sr_table[self.last_state, :, self.last_action] = self.sr_table[self.last_state, :, self.last_action] * (1 - self.alpha) + (self.sr_table[state].max(axis=1) * self.gamma + hot) * self.alpha

    def update_table(self, state, reward):
        # update reward table
        self.rewards[state] = reward
        # update sr table (usign cupy)
        new_sr_row = self.sr_table[self.last_state, :, self.last_action]
        new_sr_row *= (1 - self.alpha)
        alpha_term = self.sr_table[state].max(axis=1)
        alpha_term *= self.alpha * self.gamma
        alpha_term[state] += self.alpha # this is equivalent for hot vector addition but faster
        new_sr_row += alpha_term
        self.sr_table[self.last_state, :, self.last_action] = new_sr_row

    # def update_table(self, state, reward):
    #     # update reward table
    #     self.rewards[state] = reward
    #     # update sr table (usign cupy)
    #     new_sr_row = self.sr_table[self.last_state, :, self.last_action]
    #     new_sr_row *= (1 - self.alpha)
    #     alpha_term = np.array(
    #         [max(self.sr_table[state, i]) for i in range(self.table_size)]
    #     )
    #     alpha_term *= self.alpha * self.gamma
    #     alpha_term[state] += self.alpha # this is equivalent for hot vector addition but faster
    #     new_sr_row += alpha_term
    #     self.sr_table[self.last_state, :, self.last_action] = new_sr_row

    # @jit(nopython=True)
    # def numba_func(sr_row, sr_mat, alpha, gamma):
    #     sr_row *= (1 - alpha)
    #     alpha_term = np.array([sr_mat[i].max() for i in range(sr_mat.shape[0])])
    #     alpha_term *= alpha * gamma
    #     alpha_term += alpha
    #     sr_row += alpha_term
    #     return sr_row
    
    # def update_table(self, state, reward):
    #     # update reward table
    #     self.rewards[state] = reward
    #     # update sr table (using numba)
    #     new_sr_row = SRAgent.numba_func(self.sr_table[self.last_state, :, self.last_action], self.sr_table[state], self.alpha, self.gamma)
    #     self.sr_table[self.last_state, :, self.last_action] = new_sr_row

    def show_qtable(self):
        # q_tableの可視化
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
    
    def show_qtable_max(self):
        # q_tableの可視化. maxのみ
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