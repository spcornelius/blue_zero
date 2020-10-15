from typing import Tuple

import torch
import numpy as np
from gym import Env
from torch import Tensor

from blue_zero.config import Status
from blue_zero.clusters import find_clusters
from blue_zero.gui import BlueGUI
import blue_zero.config as cfg

__all__ = []
__all__.extend([
    'Blue'
])


class Blue(Env):

    def __init__(self, state: '2D array_like',
                 with_gui: bool = False,
                 screen_size: Tuple[int, int] = cfg.screen_size):
        super().__init__()

        state = torch.as_tensor(state, dtype=torch.uint8)
        assert state.ndim == 2
        assert np.isin(state.cpu().numpy(), Status).all()
        self.state = state
        self._r_norm = np.prod(self.state.shape)

        self.with_gui = with_gui

        self.states = []
        self.actions = []
        self.rewards = []
        self.max_non_lcc_size = 0
        self.steps_taken = 0
        self.r = 0.0

        # keep copy of initial state for resetting
        self._state_orig = self.state.detach().clone()
        self._game_over = False

        if self.with_gui:
            self.gui = BlueGUI(self.state.shape,
                               screen_size=screen_size)

        self.update()

    def _get_clusters(self):
        # get clusters, their sizes, and mapping of grid cell to cluster
        s = self.state.cpu().numpy()
        not_blocked = (s == Status.alive) | (s == Status.dead)
        cluster_map = find_clusters(not_blocked)
        clusters, sizes = np.unique(cluster_map[cluster_map != 0],
                                    return_counts=True)
        # remap clusters to start at 0
        clusters -= 1
        cluster_map -= 1
        return clusters, sizes, cluster_map

    @classmethod
    def from_random(cls, size: tuple, p: float, **kwargs):
        if not (0 <= p <= 1):
            raise ValueError("Fill probability p must be between 0 and 1.")
        return cls(np.random.uniform(size=size) < p, **kwargs)

    @property
    def done(self) -> bool:
        return self._game_over

    @property
    def sol_size(self):
        return int(np.abs(self.r) * self._r_norm)

    def reward(self, action) -> float:
        return -1.0 / self._r_norm

    @property
    def action_space(self) -> Tensor:
        s = self.state
        return ((s == Status.alive) | (s == Status.dead)).nonzero()

    def update(self) -> None:
        clusters, sizes, cluster_map, = self._get_clusters()

        if len(clusters) > 1:
            self.max_non_lcc_size = max(self.max_non_lcc_size,
                                        np.sort(sizes)[::-1][1])

        small = torch.from_numpy(
            sizes[clusters[cluster_map]]) <= self.max_non_lcc_size
        small = small.to(device=self.state.device)

        self.state[(self.state == Status.alive) & small] = Status.dead
        self._game_over = not (self.state == Status.alive).any().item()
        self.render()

    def render(self):
        if self.with_gui:
            self.gui.draw_board(self.state)

    def step(self, ij) -> Tuple[Tensor, float, bool, None]:
        i, j = tuple(ij)
        if ((self.state[i, j] == Status.wall) |
            (self.state[i, j] == Status.attacked)).item():
            raise ValueError(f"Position ({i}, {j}) is an invalid move.")

        # record current status in history
        self.states.append(self.state.detach().clone())
        self.actions.append(ij)
        self.rewards.append(self.r)

        # get reward and update state
        dr = self.reward(ij)
        self.state[i, j] = Status.attacked
        self.update()
        self.r += dr
        self.steps_taken += 1

        # recompute clusters and check termination
        return self.state, dr, self._game_over, None

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.r = 0.0
        self.steps_taken = 0
        self.max_non_lcc_size = 0
        self.state = self._state_orig.detach().clone()
        self.update()

    def to(self, *args, **kwargs):
        self.state = self.state.to(*args, **kwargs)
        for s_prev in self.states:
            s_prev.to(*args, **kwargs)
        for a in self.actions:
            try:
                a.to(*args, **kwargs)
            except AttributeError:
                continue
        self._state_orig = self._state_orig.to(*args, **kwargs)
        return self
