from . import BM


class BM10(BM):
    def __init__(self, **kwargs):
        super().__init__(radius=500, hole_radius=20, near_hole_radius=80, dist_centor_to_hole=400, near_centor_radius=80, **kwargs)