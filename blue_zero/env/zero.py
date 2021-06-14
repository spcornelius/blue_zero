import numpy as np

from blue_zero.clusters import find_clusters
from blue_zero.config import Status
from blue_zero.env.base import BlueEnv

__all__ = []
__all__.extend([
    'BlueMode0'
])


class BlueMode0(BlueEnv, id='zero'):

    def __init__(self, *args, **kwargs):
        self.max_non_lcc_size = 0
        super().__init__(*args, **kwargs)

    def update(self):
        s = self.state
        not_blocked = np.logical_or(s[Status.alive], s[Status.dead])
        labels, cluster_sizes = find_clusters(not_blocked)

        # note: cluster 0 is an artificial cluster corresponding to the wall
        # squares. cluster_sizes[0] will always be 0.
        if len(cluster_sizes) > 2:
            self.max_non_lcc_size = max(self.max_non_lcc_size,
                                        np.sort(cluster_sizes)[-2])

        idx = (s[Status.alive]) & \
              (cluster_sizes[labels] <= self.max_non_lcc_size)
        s[:, idx] = False
        s[Status.dead, idx] = True
        self._game_over = np.all(cluster_sizes <= self.max_non_lcc_size)
        super().update()

    def reset(self):
        self.max_non_lcc_size = 0
        super().reset()
