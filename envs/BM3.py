from envs.BM import BM


class BM3(BM):
    def __init__(self):
        super().__init__(radius=150, hole_radius=10, dist_centor_to_hole=20, goal_index=0)