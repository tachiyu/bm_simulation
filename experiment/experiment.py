import numpy as np
import matplotlib.pyplot as plt

from envs import BM

class Experiment:
    def __init__(self, name:str, savedir:str):
        self.bm = []
        self.name = name
        self.savedir = savedir

    def append(self, bm:BM):
        self.bm.append(bm)

    def get(self, index:int):
        return self.bm[index]
    
    def travel_distance(self, trajecotry:list):
        return len(trajecotry) - 1

    def no_of_error(self, trajectory:list, bm:BM):
        cnt = 0
        holes = bm.holes
        goal_index = bm.goal_index
        near_hole_radius = bm.near_hole_radius
        N_HOLES = bm.N_HOLES
        for x, y in trajectory:
            for i in range(N_HOLES):
                if i != goal_index and BM.norm(x, y, holes[i][0], holes[i][1]) < near_hole_radius:
                    cnt += 1
                    break
        return cnt
    
    def plot_conventionals(self):
        # plot travel distance and no of error x axis: trial, y axis: travel distance or no of error (median of agents)
        travel_distanses = [[self.travel_distance(state_history) for state_history in bm.all_state_history] for bm in self.bm]
        median_travel_distanses = [np.median(between_agents) for between_agents in zip(*travel_distanses)]
        no_of_errors = [[self.no_of_error(state_history, bm) for state_history in bm.all_state_history] for bm in self.bm]
        median_no_of_errors = [np.median(between_agents) for between_agents in zip(*no_of_errors)]
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(median_travel_distanses)
        ax[0].set_title("travel distance")
        ax[1].plot(median_no_of_errors)
        ax[1].set_title("no of error")
        plt.savefig(f"{self.savedir}/{self.name}.png")

    def strategy(self, trajectory:list, bm:BM):
        holes = bm.holes
        goal_index = bm.goal_index
        box_size = bm.radius*2
        def quadrant(state):
            x, y = state
            if x < box_size/2 and y < box_size/2:
                return 0
            elif x >= box_size/2 and y < box_size/2:
                return 1
            elif x >= box_size/2 and y >= box_size/2:
                return 2
            elif x < box_size/2 and y >= box_size/2:
                return 3
        quadrant_trajectory = [quadrant(state) for state in trajectory]
        cross = 0
        for i in range(len(quadrant_trajectory)-1):
            if quadrant_trajectory[i] != quadrant_trajectory[i+1]:
                cross += 1
        no_of_error = self.no_of_error(trajectory, bm)
        if cross == 0 and no_of_error < 3:
            return "spatial"
        elif cross < 3:
            return "serial"
        else:
            return "random"
        
    def plot_strategy(self):
        # cumulative bar plot of strategy x axis: trial, y axis: cumulative number of agents
        strategies = [[self.strategy(state_history, bm) for state_history in bm.all_state_history] for bm in self.bm]
        spatial = [strategy.count("spatial") for strategy in zip(*strategies)]
        serial = [strategy.count("serial") for strategy in zip(*strategies)]
        random = [strategy.count("random") for strategy in zip(*strategies)]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.bar(range(len(random)), random, label="random")
        ax.bar(range(len(serial)), serial, bottom=random, label="serial")
        ax.bar(range(len(spatial)), spatial, bottom=[s + r for s, r in zip(serial, random)], label="spatial")
        ax.set_title("strategy")
        ax.legend()
        plt.savefig(f"{self.savedir}/{self.name}_strategy.png")