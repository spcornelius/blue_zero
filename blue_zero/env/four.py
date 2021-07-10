import numpy as np
from scipy.signal import convolve2d

from blue_zero.clusters import find_clusters
from blue_zero.config import Status
from blue_zero.env.base import BlueEnv

__all__ = []
__all__.extend([
    'BlueMode4'
])


class BlueMode4(BlueEnv, id=4):
    """ This is the noodle mode.
    A component turns blue if the ratio of black contacts to size
    drops below K (the noodle dimension) """

    touch_filter = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def __init__(self, *args, d=2.0, **kwargs):
        """ K is the noodle dimension, the ratio of boundary contacts
        to cluster size below which the cluster becomes blue """
        self.d = d
        super().__init__(*args, **kwargs)

    def update(self):
        s = self.state
        blocked = np.logical_or(s[Status.wall], s[Status.attacked])
        labels, sizes = find_clusters(~blocked)

        # all sites are boundary sites or `not_blocked so we negate`
        boundary_contacts = convolve2d(blocked, self.touch_filter,
                                       mode="same", boundary="fill",
                                       fillvalue=1)
        boundary_contacts *= ~blocked

        # count total contacts per cluster (can this be vectorized?)
        # cluster_contacts = np.zeros_like(sizes)
        # for i in range(self.state.shape[0]):
        #     for j in range(self.state.shape[1]):
        #         cluster_contacts[labels[i, j]] += boundary_contacts[i, j]
        # nz_sizes = sizes[labels] + (sizes[labels] == 0).astype(int)
        perimeters = np.bincount(labels.flat, weights=boundary_contacts.flat)

        sizes[0] = 1
        idx = ((perimeters / sizes) >= self.d)[labels]
        self.state[:, idx] = False
        self.state[Status.dead, idx] = True
        self._game_over = ~np.any(self.state[Status.alive])
        super().update()
