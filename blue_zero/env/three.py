import numpy as np

from blue_zero.clusters import find_clusters
from blue_zero.config import Status
from blue_zero.env.base import BlueEnv

__all__ = []
__all__.extend([
    'BlueMode3'
])


class BlueMode3(BlueEnv, id=3):

    def __init__(self, *args, direction: str = 'horizontal', **kwargs):
        direction = direction.lower()
        if direction not in ['horizontal', 'vertical', 'both']:
            raise ValueError(
                "direction must be one of ['horizontal', 'vertical', 'both']")
        self.__direction = direction
        self.__direction = direction
        super().__init__(*args, **kwargs)

    @property
    def direction(self):
        return self.__direction

    def _get_spanning_clusters(self, clusters):
        left = set(clusters[:, 0])
        right = set(clusters[:, -1])
        top = set(clusters[0, :])
        bottom = set(clusters[-1, :])
        if self.direction == 'horizontal':
            sc = left & right
        elif self.direction == 'vertical':
            sc = top & bottom
        elif self.direction == 'both':
            sc = (left & right) | (top & bottom)
        sc.discard(0)
        return sc

    def update(self):
        s = self.state
        not_blocked = np.logical_or(s[Status.alive], s[Status.dead])
        labels, cluster_sizes = find_clusters(not_blocked)
        spanning_clusters = self._get_spanning_clusters(labels)

        idx = np.isin(labels, list(spanning_clusters))
        s[:, not_blocked] = False
        s[Status.alive, idx] = True
        self._game_over = not spanning_clusters
        super().update()
