import numpy as np

from blue_zero.clusters import find_clusters
from blue_zero.config import Status
from blue_zero.env.base import BlueBase

__all__ = []
__all__.extend([
    'BlueMode3'
])


class BlueMode3(BlueBase):

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
            return left & right
        elif self.direction == 'vertical':
            return top & bottom
        elif self.direction == 'both':
            return (left & right) | (top & bottom)

    def update(self):
        s = self.state
        not_blocked = np.logical_or(s == Status.alive, s == Status.dead)
        clusters, sizes = find_clusters(not_blocked)
        spanning_clusters = self._get_spanning_clusters(clusters)
        self.state[not_blocked] = Status.dead
        idx = np.isin(clusters, list(spanning_clusters))
        self.state[idx] = Status.alive
        self._game_over = idx.sum() == 0
        super().update()
