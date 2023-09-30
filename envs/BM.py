import numpy as np
import matplotlib.pyplot as plt

class BM:
    N_HOLES = 12
    STEP_SIZE = 5

    def __init__(self, radius:int, hole_radius:int, near_hole_radius:int, dist_centor_to_hole:int, near_centor_radius:int, episode:int=-1, n_actions:int=8, goal_index:int=-1, max_step:int=100000, habituation_max_step:int=100000):
        # 環境設定
        self.radius = radius 
        self.env_size = (radius*2, radius*2)
        self.hole_radius = hole_radius
        self.near_hole_radius = near_hole_radius
        self.near_centor_radius = near_centor_radius
        self.max_step = max_step
        self.habituation_max_step = habituation_max_step
        self.goal_index = np.random.randint(self.N_HOLES) if goal_index == -1 else goal_index
        self.holes = [(radius + dist_centor_to_hole*np.cos(2*np.pi*i/self.N_HOLES), radius + dist_centor_to_hole*np.sin(2*np.pi*i/self.N_HOLES)) for i in range(self.N_HOLES)]
        # actionの定義
        self.n_actions = n_actions
        self.action_to_move = [(self.STEP_SIZE*np.cos(2*np.pi*i/n_actions), self.STEP_SIZE*np.sin(2*np.pi*i/n_actions)) for i in range(n_actions)]
        # キャッシュ
        self.avalable_state_cache = set()
        self.unabalable_state_cache = set()
        self.step_count = 0
        # episodeの番号。もし-1ならhabituation
        self.episode = episode 

        # 行動の履歴
        self.action_history = []
        self.all_action_history = []
        # 状態の履歴
        self.state_history = []
        self.all_state_history = []
        # 備考の履歴
        self.info_history = []
        self.all_info_history = []
        # 状態の初期化
        self.reset(episode=episode)

    def reset(self, episode:int=-1):
        self.state = (self.radius, self.radius)
        self.episode = episode
        self.step_count = 0
        if episode != -1:
            if self.action_history != []:
                self.all_action_history.append(self.action_history)
            if self.state_history != []:
                self.all_state_history.append(self.state_history)
            if self.info_history != []:
                self.all_info_history.append(self.info_history)
        self.action_history = []
        self.state_history = []
        self.info_history = []
        return {"state": self.state,
                "next_actions": self.get_next_actions(), 
                "reward": 0, 
                "done": False,
                "info": ""}
    
    def move(self, action:int):
        return (self.state[0] + self.action_to_move[action][0], self.state[1] + self.action_to_move[action][1])
    
    def check_next_state(self, next_state:tuple):
        # check if next_state is in cache
        if next_state in self.avalable_state_cache:
            return True
        elif next_state in self.unabalable_state_cache:
            return False
        else: 
            x, y = next_state
            # check if next_state is out of bounds
            if BM.is_distance_less_than(x, y, self.radius, self.radius, self.radius):
                self.avalable_state_cache.add(next_state)
                return True
            else :
                self.unabalable_state_cache.add(next_state)
                return False
    
    def get_next_actions(self):
        return [action for action in range(self.n_actions) if self.check_next_state(self.move(action))]
    
    def step(self, action):
        self.state = self.move(action)
        x, y = self.state
        info = ""
        reward = 0
        done = False

        if self.episode != -1: # check not habituation
            if BM.is_distance_less_than(x, y, self.radius, self.radius, self.near_centor_radius): # check if near centor
                info = "near_centor"
            else: 
                for i in range(self.N_HOLES):
                    if BM.is_distance_less_than(x, y, self.holes[i][0], self.holes[i][1], self.near_hole_radius): # check if near hole
                        if i == self.goal_index: # check if goal
                            if BM.is_distance_less_than(x, y, self.holes[i][0], self.holes[i][1], self.hole_radius): # check if in goal hole
                                reward = 1
                                done = True
                                info = "goal"
                            else: # near goal hole
                                info = "near_goal"
                        else: # near dammy hole
                            info = "near_dummy"
                        break

            self.action_history.append(action)
            self.state_history.append(self.state)
            self.info_history.append(info)

        self.step_count += 1
        if self.episode == -1:
            if self.step_count >= self.habituation_max_step:
                done = True
        else:
            if self.step_count >= self.max_step:
                done = True

        return {"state": self.state, 
                "next_actions": self.get_next_actions(),
                "reward": reward, 
                "done": done,
                "info": info}
    
    def render(self):
        # render the maze
        plt.gca().cla()
        plt.gca().set_xlim(0, self.radius*2)
        plt.gca().set_ylim(0, self.radius*2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().add_artist(plt.Circle((self.radius, self.radius), self.radius, color='black', fill=False))
        plt.gca().set_title(f"episode: {self.episode}, step: {self.step_count}")
        for i in range(self.N_HOLES):
            plt.gca().add_artist(plt.Circle((self.holes[i][0], self.holes[i][1]), self.hole_radius, color='black', fill=True))
        plt.gca().add_artist(plt.Circle((self.holes[self.goal_index][0], self.holes[self.goal_index][1]), self.hole_radius, color='red', fill=True))

        # render the agent
        plt.gca().add_artist(plt.Circle((self.state[0], self.state[1]), 1, color='blue', fill=True))
        plt.pause(0.1)

    @staticmethod
    def is_distance_less_than(x1, y1, x2, y2, dist):
        return (x1-x2)**2 + (y1-y2)**2 < dist**2
