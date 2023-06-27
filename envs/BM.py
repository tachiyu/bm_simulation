import time
from gymnasium import spaces, Env
import numpy as np
import matplotlib.pyplot as plt

class BM(Env):
    N_HOLES = 12

    def __init__(self, radius:int, hole_radius:int, dist_centor_to_hole:int, goal_index:int):
        self.radius = radius
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([0,0]), high=np.array([radius*2, radius*2]), dtype=np.uint8)
        self.goal_index = goal_index
        self.hole_radius = hole_radius
        self.holes = [[radius + dist_centor_to_hole*np.cos(2*np.pi*i/self.N_HOLES), radius + dist_centor_to_hole*np.sin(2*np.pi*i/self.N_HOLES)] for i in range(self.N_HOLES)]
        self.reset()

    def reset(self):
        self.state = np.array([self.radius, self.radius])
        self.done = False
        return self.state

    def step(self, action):
        done = False
        reward = 0
        info = set()

        if action == 0:
            next_state = self.state + np.array([0, 1])
        elif action == 1:
            next_state = self.state + np.array([0, -1])
        elif action == 2:
            next_state = self.state + np.array([1, 0])
        elif action == 3:
            next_state = self.state + np.array([-1, 0])
        else:
            raise ValueError("Invalid action")
        
        # check if next_state is out of bounds
        if np.linalg.norm(next_state - np.array([self.radius, self.radius])) > self.radius:
            next_state = self.state
            info.add("out of bounds")
        
        # if next_state is goal, reward = 1, else reward = 0 and revert to previous state
        for i in range(self.N_HOLES):
            if np.linalg.norm(next_state - np.array(self.holes[i])) < self.hole_radius:
                if i == self.goal_index:
                    done = True
                    reward = 1
                    info.add("goal")
                else:
                    next_state = self.state
                    info.add("non-goal hole")

        self.state = next_state

        return self.state, reward, done, info
    
    def render(self):
        # render the maze
        plt.gca().cla()
        plt.gca().set_xlim(0, self.radius*2)
        plt.gca().set_ylim(0, self.radius*2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().add_artist(plt.Circle((self.radius, self.radius), self.radius, color='black', fill=False))
        for i in range(self.N_HOLES):
            plt.gca().add_artist(plt.Circle((self.holes[i][0], self.holes[i][1]), self.hole_radius, color='black', fill=True))
        plt.gca().add_artist(plt.Circle((self.holes[self.goal_index][0], self.holes[self.goal_index][1]), self.hole_radius, color='red', fill=True))

        # render the agent
        plt.gca().add_artist(plt.Circle((self.state[0], self.state[1]), 1, color='blue', fill=True))
        plt.pause(0.1)
