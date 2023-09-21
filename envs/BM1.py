from . import BM


class BM1(BM):
    def __init__(self, **kwargs):
        super().__init__(radius=50, hole_radius=2, near_hole_radius=8, dist_centor_to_hole=40, **kwargs)