from . import BM


class BM20(BM):
    def __init__(self, **kwargs):
        super().__init__(radius=1000, hole_radius=40, near_hole_radius=160, dist_centor_to_hole=800, near_centor_radius=160, **kwargs)