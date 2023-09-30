from . import BM


class BM3(BM):
    def __init__(self, **kwargs):
        super().__init__(radius=150, hole_radius=6, near_hole_radius=27, dist_centor_to_hole=140, near_centor_radius=27, **kwargs)