from envs.BM import BM


class BM1(BM):
    def __init__(self, goal_index):
        super().__init__(radius=50, hole_radius=2, dist_centor_to_hole=40, goal_index=goal_index)