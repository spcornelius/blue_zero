import numpy as np
from scipy.signal import convolve2d

from blue_zero.clusters import find_clusters
from blue_zero.config import Status
from blue_zero.env.base import BlueBase

__all__ = []
__all__.extend([
    'BlueMode4'
])


class BlueMode4(BlueBase):
    '''
    This is the noodle mode. 
    A component turns blue if the ratio of black contacts to size
    drops below K (the noodle dimension)
    '''


    def __init__(self, *args, K=2.0,**kwargs):
        '''
        K is the noodle dimension, the ratio of boundary contacts
        to cluster size below which the cluster becomes blue
        '''
        self.K = K
        super().__init__(*args, **kwargs)
        

    def update(self):
        s = self.state
        not_blocked = np.logical_or(s == Status.alive, s == Status.dead)
        clusters, sizes = find_clusters(not_blocked)    
        touching_filter = np.array([[0,1,0],[1,0,1],[0,1,0]])
        #all sites are boundary sites or `not_blocked so we negate`
        boundary_contacts = convolve2d(~not_blocked, touching_filter, 
                        mode="same",boundary="fill", fillvalue=1)
        boundary_contacts *= not_blocked
        #count total contacts per cluster (can this be vectorized?)
        cluster_contacts = np.zeros_like(sizes)
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                cluster_contacts[clusters[i,j]]+=boundary_contacts[i,j]
        nonzerosizes = sizes[clusters] + (sizes[clusters] == 0).astype(int)
        
        idx = (cluster_contacts[clusters] / nonzerosizes >= self.K)
        self.state[idx] = Status.dead
        self._game_over = np.all(self.state != Status.alive)
        super().update()