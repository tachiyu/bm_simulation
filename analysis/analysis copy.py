import numpy as np
import matplotlib.pyplot as plt

from envs import BM

class Analysis:
    def __init__(self, name:str, savedir:str):
        self.bm = []
        self.bm_group = []
        self.name = name
        self.savedir = savedir

    def append(self, bm:BM):
        self.bm.append(bm)

    def append(self, bm_group:list[BM], group_name:str):
        self.bm_group.append([group_name, bm_group])

    def get(self, index:int):
        return self.bm[index]
    
    def travel_distance(self, trajectory:list):
        return len(trajectory) - 1

    def no_of_error(self, infos:list):
        return infos.count("near_dummy")
    
    def plot_conventionals(self):
        # plot travel distance and no of error x axis: trial, y axis: travel distance or no of error (median of agents)
        travel_distanses = [[self.travel_distance(state_history) for state_history in bm.all_state_history] for bm in self.bm]
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

    def plot_conventionals_group(self, show_error=True, limit=None):
        # plot travel distance and no of error x axis: trial, y axis: travel distance or no of error (median of agents)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title("travel distance")
        ax[1].set_title("no of error")
        for group_name, bm_group in self.bm_group:
            travel_distanses = [[self.travel_distance(state_history) for state_history in bm.all_state_history] for bm in bm_group]
            q1_travel_distanses, q2_travel_distanses, q3_travel_distanses = zip(*[np.percentile(between_agents, [25, 50, 75]) for between_agents in zip(*travel_distanses)])
            no_of_errors = [[self.no_of_error(info_history) for info_history in bm.all_info_history] for bm in bm_group]
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

    def strategy(self, trajectory:list, infos:list, bm:BM):
        harfed_box_size = bm.radius*2 / 2
        def quadrant(state):
            return (state[0] < harfed_box_size) + (state[1] < harfed_box_size) * 2
        
        quadrant_trajectory = [quadrant(state) for state, info in zip(trajectory, infos) if info != "near_centor"]

        cross = 0
        for i in range(len(quadrant_trajectory)-1):
            if quadrant_trajectory[i] != quadrant_trajectory[i+1]:
                cross += 1

        no_of_error = self.no_of_error(infos)

        if cross == 0 and no_of_error < 3:
            return "spatial"
        elif cross < 3:
            return "serial"
        else:
            return "random"
        
    def plot_strategy(self):
        # cumulative bar plot of strategy x axis: trial, y axis: cumulative number of agents
        strategies = [[self.strategy(state_history, infos, bm) for state_history, infos in zip(bm.all_state_history, bm.all_info_history)] for bm in self.bm]
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