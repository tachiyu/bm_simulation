import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

from agents import Agent
from envs import BM

# 以下のコードは博論においては未使用
# Conventional analysisやStrategy analysisにおいて使用を想定していたが、実際には使用されていない

class Analysis:
    def __init__(self, name:str="", savedir:str=""):
        self.bm: list[BM] = [] 
        self.bm_group: list[tuple[str, list[BM]]] = []
        self.name = name
        self.savedir = savedir

    def append(self, bm:BM):
        self.bm.append(bm)

    def append(self, bm_group:list[BM], group_name:str):
        self.bm_group.append((group_name, bm_group))

    def get(self, index:int):
        return self.bm[index]
    
    def travel_distance(self, bm:BM, trial_index:int):
        return len(bm.trajectory_list[trial_index]) - 1

    def no_of_error(self, bm:BM, trial_index:int):
        trajectory = bm.trajectory_list[trial_index]
        error_holes_index = [i for i in range(bm.N_HOLES) if i != bm.goal_index]
        error_cnt = 0
        for x, y in trajectory:
            for i in error_holes_index:
                if bm.is_near_hole((x,y), i):
                    error_cnt += 1
        return error_cnt

    def quadrant_cross(self, bm:BM, trial_index:int):
        trajectory = bm.trajectory_list[trial_index]
        cross_cnt = 0
        current_quadrant = None
        for x, y in trajectory:
            if bm.is_near_centor((x,y)):
                continue
            if current_quadrant is None:
                current_quadrant = x < bm.radius + 2 * (y < bm.radius)
            else:
                next_quadrant = x < bm.radius + 2 * (y < bm.radius)
                if current_quadrant != next_quadrant:
                    cross_cnt += 1
                current_quadrant = next_quadrant
        return cross_cnt
    
    def plot_conventionals(self):
        # plot travel distance and no of error x axis: trial, y axis: travel distance or no of error (median of agents)
        travel_distanses = [[self.travel_distance(bm, i) for i in range(bm.training_days)] for bm in self.bm]
        q1_travel_distanses, q2_travel_distanses, q3_travel_distanses = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*travel_distanses)])

        no_of_errors = [[self.no_of_error(info_history) for info_history in bm.all_info_history] for bm in self.bm]
        q1_no_of_errors, q2_no_of_errors, q3_no_of_errors = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*no_of_errors)])

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(q2_travel_distanses)
        ax[0].fill_between(range(len(q2_travel_distanses)), q1_travel_distanses, q3_travel_distanses, alpha=0.5)
        ax[0].set_title("travel distance")
        ax[1].plot(q2_no_of_errors)
        ax[1].fill_between(range(len(q2_no_of_errors)), q1_no_of_errors, q3_no_of_errors, alpha=0.5)
        ax[1].set_title("no of error")
        plt.savefig(f"{self.savedir}/{self.name}.png")

    def plot_travel_distance_group(self, show_error=True, limit=None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.set_title("travel distance")
        for group_name, bm_group in self.bm_group:
            travel_distanses = [[self.travel_distance(bm, i) for i in range(bm.training_days)] for bm in bm_group]
            q1_travel_distanses, q2_travel_distanses, q3_travel_distanses = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*travel_distanses)])
            ax.plot(q2_travel_distanses, label=group_name)
            if show_error:
                ax.fill_between(range(len(q2_travel_distanses)), q1_travel_distanses, q3_travel_distanses, alpha=0.5)
        ax.legend()
        if limit is not None:
            ax.set_ylim(0, limit)
        plt.savefig(f"{self.savedir}/{self.name}_travel_distance.png")

    def plot_conventionals_group(self, show_error=True, limit=None):
        # plot travel distance and no of error x axis: trial, y axis: travel distance or no of error (median of agents)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title("travel distance")
        ax[1].set_title("no of error")
        for group_name, bm_group in self.bm_group:
            travel_distanses = [[self.travel_distance(bm, i) for i in range(bm.training_days)] for bm in bm_group]
            q1_travel_distanses, q2_travel_distanses, q3_travel_distanses = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*travel_distanses)])
            no_of_errors = [[self.no_of_error(bm, i) for i in range(bm.training_days)] for bm in bm_group]
            q1_no_of_errors, q2_no_of_errors, q3_no_of_errors = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*no_of_errors)])
            ax[0].plot(q2_travel_distanses, label=group_name)
            ax[1].plot(q2_no_of_errors, label=group_name)
            if show_error:
                ax[0].fill_between(range(len(q2_travel_distanses)), q1_travel_distanses, q3_travel_distanses, alpha=0.5)
                ax[1].fill_between(range(len(q2_no_of_errors)), q1_no_of_errors, q3_no_of_errors, alpha=0.5)
        ax[0].legend()
        ax[1].legend()
        if limit is not None:
            ax[0].set_ylim(0, limit[0])
            ax[1].set_ylim(0, limit[1])
        plt.savefig(f"{self.savedir}/{self.name}.png")

    def strategy(self, bm:BM, trial_index:int):
        no_of_error = self.no_of_error(bm, trial_index)
        cross = self.quadrant_cross(bm, trial_index)
        if cross == 0 and no_of_error < 3:
            return "spatial"
        elif cross < 3:
            return "serial"
        else:
            return "random"
        
    def plot_strategy(self):
        # cumulative bar plot of strategy x axis: trial, y axis: cumulative number of agents
        strategies = [[self.strategy(bm, i) for i in range(bm.training_days)] for bm in self.bm]
        spatial = [strategy.count("spatial") for strategy in zip(*strategies)]
        serial = [strategy.count("serial") for strategy in zip(*strategies)]
        random = [strategy.count("random") for strategy in zip(*strategies)]
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.bar(range(len(spatial)), spatial, bottom=[s + r for s, r in zip(serial, random)], label="spatial")
        ax.bar(range(len(serial)), serial, bottom=random, label="serial")
        ax.bar(range(len(random)), random, label="random")
        ax.set_title("strategy")
        ax.legend()
        plt.savefig(f"{self.savedir}/{self.name}_strategy.png")

    def read_travel_distances(self, bm:BM):
        return [self.travel_distance(bm, i) for i in range(bm.training_days)]
    
    def exstract_travel_distance(self, path):
        with open(path, "rb") as f:
            print(f"loading {path}")
            bms: list[BM] = pickle.load(f)
        print("extracting travel distance...")
        tds = [self.read_travel_distances(bm) for bm in bms]
        npath = f"results/travel_dist/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(npath), exist_ok=True)
        with open(npath, "wb") as f:
            print(f"saving {npath}")
            pickle.dump(tds, f)

    def exstract_travel_distance_trajOff(self, path, savedir):
        with open(path, "rb") as f:
            print(f"loading {path}")
            bms: list[BM]|list[tuple[Agent, BM]] = pickle.load(f)
        print("extracting travel distance...")
        if "withAgent" in path.name:
            tds = [bm[1].step_count_list for bm in bms]
        else:
            tds = [bm.step_count_list for bm in bms]
        npath = f"{savedir}/{os.path.basename(path)}"
        os.makedirs(os.path.dirname(npath), exist_ok=True)
        with open(npath, "wb") as f:
            print(f"saving {npath}")
            pickle.dump(tds, f)