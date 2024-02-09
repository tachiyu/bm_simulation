import os
import pickle
import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# すべてのBM環境の基底クラス
class BM:
    N_HOLES = 12
    STEP_SIZE = 5
    def __init__(
            self, radius:int, # 環境の半径
            hole_radius:int, # 穴の半径
            near_hole_radius:int,  # 穴の近くの半径
            dist_centor_to_hole:int, # 中心から穴までの距離
            near_centor_radius:int,  # 中心の近くの半径
            episode:int=-1,     # episode (試行) の番号。もし-1ならhabituation
            n_actions:int=8,  # actionの数
            goal_index:int=-1,  # 目標の穴のindex
            max_step:int=100000,  # 最大ステップ数 (試行を打ち切る)
            habituation_max_step:int=100000,  # habituationの最大ステップ数
            memorize_trajectory:bool=True # 状態の履歴を記録するかどうか (実行速度とメモリ使用量に注意すること)
        ):
        # 環境設定
        self.radius = radius # 環境の半径
        self.env_size = (radius*2, radius*2) # 環境のサイズ
        self.hole_radius = hole_radius # 穴の半径
        self.near_hole_radius = near_hole_radius # 穴の近くの半径
        self.near_centor_radius = near_centor_radius # 中心の近くの半径
        self.max_step = max_step # 最大ステップ数 (試行を打ち切る)
        self.habituation_max_step = habituation_max_step # habituationの最大ステップ数
        self.goal_index = np.random.randint(self.N_HOLES) if goal_index == -1 else goal_index # 目標の穴のindex
        self.holes = [(dist_centor_to_hole*np.cos(2*np.pi*i/self.N_HOLES), dist_centor_to_hole*np.sin(2*np.pi*i/self.N_HOLES)) for i in range(self.N_HOLES)] # 穴の位置
        self.n_actions = n_actions # actionの数
        self.action_to_move = [(self.STEP_SIZE*np.cos(2*np.pi*i/n_actions), self.STEP_SIZE*np.sin(2*np.pi*i/n_actions)) for i in range(n_actions)] # actionを実行したときの移動量
        self.training_days = 0 # trainingの日数 (初期値)
        # 状態の履歴
        self.memorize_trajectory = memorize_trajectory
        self.trajectory = []
        self.trajectory_list = []
        self.step_count_list = []
        # 状態の初期化
        self.reset(episode=episode)

    # 環境の初期化
    def reset(self, episode:int=-1):
        self.state =  (0, 0) # 初期状態は中心
        self.episode = episode # episodeの番号。もし-1ならhabituation
        self.step_count = 0 # ステップ数のカウントを初期化
        # habituationでないならtraining_daysをカウントアップ
        if episode != -1: 
            self.training_days += 1 
        self.trajectory = [] # 軌跡を初期化
        return {"state": self.state,
                "next_actions": self.get_next_actions(), 
                "reward": 0, 
                "done": False}
    
    # エージェントのactionで環境を1ステップ進める
    def step(self, action):
        self.state = self.move(action) # actionを実行
        # 軌跡を記録
        if self.memorize_trajectory:
            self.trajectory.append(self.state)
        self.step_count += 1 # ステップ数をカウントアップ
        reward, done = 0, False # rewardとdoneを初期化
        # habituationでないならrewardとdoneを計算
        if self.episode != -1: 
            # 
            if self.__is_in_hole(self.state, self.goal_index):
                reward = 1
                done = True
            elif self.step_count >= self.max_step:
                done = True
            if done:
                self.trajectory_list.append(self.trajectory)
                self.step_count_list.append(self.step_count)

        return {"state": self.state, 
                "next_actions": self.get_next_actions(),
                "reward": reward, 
                "done": done}
    
    # エージェントが現在とることのできるactionのリストを返す
    def get_next_actions(self):
        return [action for action in range(self.n_actions) 
                if self.__is_in_maze(self.move(action))]
    
    # エージェントがactionを実行したときの次の環境における状態を返す
    def move(self, action:int):
        return (self.state[0] + self.action_to_move[action][0], self.state[1] + self.action_to_move[action][1])
    
    # 穴の上にいるか
    def __is_in_hole(self, state, hole_index) -> bool:
        hole = self.holes[hole_index]
        return (state[0]-hole[0])**2 + (state[1]-hole[1])**2 < self.hole_radius**2
    
    # 環境内にいるか
    def __is_in_maze(self, state) -> bool:
        return state[0]**2 + state[1]**2 < self.radius**2
    
    # 環境を描画する
    def render(self, plt_trajectory=False, save_path=None, pause=True):
        artists = []  # このフレームのアーティストを格納するリスト
        # BMを描画
        plt.gca().cla()
        plt.gca().set_xlim(-self.radius, self.radius)
        plt.gca().set_ylim(-self.radius, self.radius)
        plt.gca().set_aspect('equal', adjustable='box')
        # 中心の円（境界）
        boundary_circle = plt.Circle((0, 0), self.radius, color='black', fill=False)
        plt.gca().add_artist(boundary_circle)
        artists.append(boundary_circle)
        # タイトルの設定
        title = plt.gca().set_title(f"episode: {self.episode}, step: {self.step_count}")
        artists.append(title)
        # 穴を描画
        for i in range(self.N_HOLES):
            hole_circle = plt.Circle((self.holes[i][0], self.holes[i][1]), self.hole_radius, color='black', fill=True)
            plt.gca().add_artist(hole_circle)
            artists.append(hole_circle)
        # 目標の穴（赤色）
        goal_circle = plt.Circle((self.holes[self.goal_index][0], self.holes[self.goal_index][1]), self.hole_radius, color='red', fill=True)
        plt.gca().add_artist(goal_circle)
        artists.append(goal_circle)
        # エージェントを描画
        agent_circle = plt.Circle((self.state[0], self.state[1]), 1, color='blue', fill=True)
        plt.gca().add_artist(agent_circle)
        artists.append(agent_circle)
        # 軌跡を描画
        if plt_trajectory:
            for i in range(len(self.trajectory) - 1):
                line, = plt.plot([self.trajectory[i][0], self.trajectory[i+1][0]], [self.trajectory[i][1], self.trajectory[i+1][1]], color='black', linewidth=0.5)
                artists.append(line)
        # ポーズと保存
        if pause:
            plt.pause(0.1)
        if save_path is not None:
            plt.savefig(save_path)

        return artists  # アーティストのリストを返す
    
    # (デバッグ用) trajectory_id番目の軌跡をfrom_stepから再生する
    def replay_trajectory(self, trajectory_id, from_step=0):
        trajectory = self.trajectory_list[trajectory_id][from_step:]
        for state in trajectory:
            self.state = state
            self.render()
