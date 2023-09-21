from gymnasium import spaces, Env
import numpy as np
import matplotlib.pyplot as plt

class BM(Env):
    N_HOLES = 12

    def __init__(self, radius:int, hole_radius:int, near_hole_radius:int, dist_centor_to_hole:int, episode:int=-1, n_actions:int=8, goal_index:int=-1, max_step:int=100000):
        # 環境の大きさ radius自体を含まないようにするために-1e-3
        self.radius = radius - 1e-3
        # 歩幅
        self.step_size = 5
        # step数の上限
        self.max_step = max_step
        # action_spaceの定義
        self.action_space = spaces.Discrete(n_actions)
        # actionに対応する移動方向
        self.action_to_move = {i: (self.step_size*np.cos(2*np.pi*i/self.action_space.n), self.step_size*np.sin(2*np.pi*i/self.action_space.n)) for i in range(self.action_space.n)}
        # observation_spaceの定義
        # 縦横の座標の最大値はradius*2。-0.1は、2*radiusを含まないようにするためのもの
        self.observation_space = spaces.Box(low=np.array([0,0]), high=np.array([radius*2, radius*2])-0.1, dtype=np.float64)
        self.observation_sup = [radius*2, radius*2]
        self.observation_inf = [0, 0]
        # ゴールの座標
        self.goal_index = np.random.randint(self.N_HOLES) if goal_index == -1 else goal_index
        self.hole_radius = hole_radius
        self.near_hole_radius = near_hole_radius
        self.holes = [[radius + dist_centor_to_hole*np.cos(2*np.pi*i/self.N_HOLES), radius + dist_centor_to_hole*np.sin(2*np.pi*i/self.N_HOLES)] for i in range(self.N_HOLES)]
        # キャッシュ
        self.avalable_state_cache = set()
        self.unabalable_state_cache = set()
        self.step_count = 0
        # episode もし-1ならhabituation
        self.episode = episode 

        # 行動の履歴
        self.action_history = []
        self.all_action_history = []
        # 状態の履歴
        self.state_history = []
        self.all_state_history = []
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
        self.action_history = []
        self.state_history = []
        return {"state": self.state,
                "next_actions": self.get_next_actions(), 
                "reward": 0, 
                "done": False,
                "info": set()}
    
    def move(self, action:int):
        return (self.state[0] + self.action_to_move[action][0], self.state[1] + self.action_to_move[action][1])
    
    def check_next_state(self, next_state:np.ndarray):
        x = next_state[0]
        y = next_state[1]

        # check if next_state is in cache
        if (x, y) in self.avalable_state_cache:
            return True
        elif (x, y) in self.unabalable_state_cache:
            return False
        else: 
            # check if next_state is out of bounds
            if BM.norm(x, y, self.radius, self.radius) > self.radius:
                self.unabalable_state_cache.add((x, y))
                return False
            else :
                self.avalable_state_cache.add((x, y))
                return True
    
    def get_next_actions(self):
        return [action for action in range(self.action_space.n) if self.check_next_state(self.move(action))]
    
    def step(self, action):
        self.state = self.move(action)
        x, y = self.state
        if self.episode != -1 and BM.norm(x, y, self.holes[self.goal_index][0], self.holes[self.goal_index][1]) < self.hole_radius:
            reward = 1
            done = True
            info = set("goal")
        else:
            # if BM.norm(x, y, self.radius, self.radius) < self.radius * 0.6:
            #     reward = -1e-3
            # elif BM.norm(x, y, self.radius, self.radius) > self.radius * 0.9:
            #     reward = -1e-3
            # else:
            #     reward = 0
            # reward = np.random.uniform(low=0, high=1e-9)
            reward = 0
            done = False
            info = set()
        self.step_count += 1
        self.action_history.append(action)
        self.state_history.append(self.state)

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
    def norm(x1, y1, x2, y2):
        return np.sqrt((x1-x2)**2 + (y1-y2)**2)
