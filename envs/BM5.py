from . import BM


class BM5(BM):
    def __init__(self, **kwargs):
        super().__init__(radius=250, hole_radius=10, near_hole_radius=40, dist_centor_to_hole=200, near_centor_radius=40, **kwargs)