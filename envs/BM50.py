from . import BM


class BM50(BM):
    def __init__(self, **kwargs):
        super().__init__(radius=2500, hole_radius=100, near_hole_radius=400, dist_centor_to_hole=2000, near_centor_radius=400, **kwargs)