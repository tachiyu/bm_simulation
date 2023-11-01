import numpy as np
from numba import jit
import matplotlib.pyplot as plt

class BM:
    N_HOLES = 12
    STEP_SIZE = 5

    def __init__(self, radius:int, hole_radius:int, near_hole_radius:int, dist_centor_to_hole:int, near_centor_radius:int, episode:int=-1, n_actions:int=8, goal_index:int=-1, max_step:int=100000, habituation_max_step:int=100000, memorize_trajectory:bool=True):
        # 環境設定
        self.radius = radius
        self.env_size = (radius*2, radius*2)
        self.hole_radius = hole_radius
        self.near_hole_radius = near_hole_radius
        self.near_centor_radius = near_centor_radius
        self.max_step = max_step
        self.habituation_max_step = habituation_max_step
        self.goal_index = np.random.randint(self.N_HOLES) if goal_index == -1 else goal_index
        self.holes = [(dist_centor_to_hole*np.cos(2*np.pi*i/self.N_HOLES), dist_centor_to_hole*np.sin(2*np.pi*i/self.N_HOLES)) for i in range(self.N_HOLES)]
        # actionの定義
        self.n_actions = n_actions
        self.action_to_move = [(self.STEP_SIZE*np.cos(2*np.pi*i/n_actions), self.STEP_SIZE*np.sin(2*np.pi*i/n_actions)) for i in range(n_actions)]
        self.training_days = 0
        # 状態の履歴
        self.memorize_trajectory = memorize_trajectory
        self.trajectory = []
        self.trajectory_list = []
        self.step_count_list = []
        # 状態の初期化
        self.reset(episode=episode)

    def reset(self, episode:int=-1):
        self.state =  (0, 0)
        self.episode = episode # episodeの番号。もし-1ならhabituation
        self.step_count = 0
        if episode != -1:
            self.training_days += 1 # trainingの日数をカウント
        self.trajectory = []
        return {"state": self.state,
                "next_actions": self.get_next_actions(), 
                "reward": 0, 
                "done": False}
    
    def move(self, action:int):
        return (self.state[0] + self.action_to_move[action][0], self.state[1] + self.action_to_move[action][1])
    
    def get_next_actions(self):
        return [action for action in range(self.n_actions) 
                if self.is_in_maze(self.move(action))]
    
    def step(self, action):
        self.state = self.move(action)
        if self.memorize_trajectory:
            self.trajectory.append(self.state)
        self.step_count += 1
        reward, done = 0, False
        if self.episode != -1: # not habituation, but training
            if self.is_near_hole(self.state, self.goal_index):
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
    
    def render(self, plt_trajectory=False, save_path=None):
        # render the maze
        plt.gca().cla()
        plt.gca().set_xlim(-self.radius, self.radius)
        plt.gca().set_ylim(-self.radius, self.radius)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().add_artist(plt.Circle((0, 0), self.radius, color='black', fill=False))
        plt.gca().set_title(f"episode: {self.episode}, step: {self.step_count}")
        for i in range(self.N_HOLES):
            plt.gca().add_artist(plt.Circle((self.holes[i][0], self.holes[i][1]), self.hole_radius, color='black', fill=True))
        plt.gca().add_artist(plt.Circle((self.holes[self.goal_index][0], self.holes[self.goal_index][1]), self.hole_radius, color='red', fill=True))
        # render the agent
        plt.gca().add_artist(plt.Circle((self.state[0], self.state[1]), 1, color='blue', fill=True))
        # render trajectory
        if plt_trajectory:
            for i in range(len(self.trajectory)-1):
                plt.plot([self.trajectory[i][0], self.trajectory[i+1][0]], [self.trajectory[i][1], self.trajectory[i+1][1]], color='black', linewidth=0.5)
        plt.pause(0.1)
        if save_path is not None:
            plt.savefig(save_path)

    def is_near_hole(self, state, hole_index) -> bool:
        hole = self.holes[hole_index]
        return (state[0]-hole[0])**2 + (state[1]-hole[1])**2 < self.near_hole_radius**2
    
    def is_in_maze(self, state) -> bool:
        return state[0]**2 + state[1]**2 < self.radius**2
    
    def is_near_centor(self, state) -> bool:
        return state[0]**2 + state[1]**2 < self.near_centor_radius**2
    

    #for post analysis
    def replay_trajectory(self, trajectory_id, from_step=0):
        trajectory = self.trajectory_list[trajectory_id][from_step:]
        for state in trajectory:
            self.state = state
            self.render()
