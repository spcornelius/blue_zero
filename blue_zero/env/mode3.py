import numpy as np

from blue_zero.clusters import find_clusters
from blue_zero.config import Status
from blue_zero.env.base import BlueBase

__all__ = []
__all__.extend([
    'BlueMode3'
])


class BlueMode3(BlueBase):

    def __init__(self, *args, direction='horizontal', **kwargs):
        self.direction = direction
        super().__init__(*args, **kwargs)

    def update(self):
        s = self.state
        not_blocked = np.logical_or(s == Status.alive, s == Status.dead)
        clusters, sizes = find_clusters(not_blocked)
        left = set(clusters[:, 0])
        right = set(clusters[:, -1])
        top = set(clusters[0, :])
        bottom = set(clusters[-1, :])
        if self.direction == 'horizontal':
            spanning_clusters = left & right
        elif self.direction == 'vertical':
            spanning_clusters = top & bottom
        elif self.direction == 'both':
            spanning_clusters = (left & right) | (top & bottom)
        spanning_clusters.discard(0)
        self.state[not_blocked] = Status.dead
        idx = np.isin(clusters, list(spanning_clusters))
        self.state[idx] = Status.alive
        self._game_over = idx.sum() == 0
        super().update()
