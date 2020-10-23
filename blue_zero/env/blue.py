from typing import Tuple

import numpy as np
from gym import Env

import blue_zero.config as cfg
from blue_zero.clusters import find_clusters
from blue_zero.config import Status
from blue_zero.gui import BlueGUI

__all__ = []
__all__.extend([
    'Blue'
])


class Blue(Env):

    def __init__(self, state: '2D array_like',
                 with_gui: bool = False,
                 screen_size: Tuple[int, int] = cfg.screen_size):
        super().__init__()

        state = np.asarray(state, dtype=np.float32)
        assert state.ndim == 2
        assert np.isin(state, Status).all()
        self.state = state
        self._r_norm = np.sqrt(np.prod(self.state.shape))

        self.with_gui = with_gui

        self.states = []
        self.actions = []
        self.rewards = []
        self.max_non_lcc_size = 0
        self.steps_taken = 0
        self.r = 0.0

        # keep copy of initial state for resetting
        self._state_orig = self.state.copy()
        self._game_over = False

        if self.with_gui:
            self.gui = BlueGUI(self.state.shape,
                               screen_size=screen_size)

        self.update()

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
    def action_space(self) -> np.ndarray:
        s = self.state
        return np.argwhere(np.logical_or(s == Status.alive,
                                         s == Status.dead))

    def update(self) -> None:
        s = self.state
        not_blocked = np.logical_or(s == Status.alive, s == Status.dead)
        labels, cluster_sizes = find_clusters(not_blocked)
        # note: cluster 0 is an artificial cluster corresponding to the wall
        # squares. cluster_sizes[0] will always be 0.
        if len(cluster_sizes) > 2:
            self.max_non_lcc_size = max(self.max_non_lcc_size,
                                        np.sort(cluster_sizes)[-2])

        idx = (s == Status.alive) & \
              (cluster_sizes[labels] <= self.max_non_lcc_size)
        self.state[idx] = Status.dead
        self._game_over = np.all(cluster_sizes <= self.max_non_lcc_size)
        self.render()

    def render(self):
        if self.with_gui:
            self.gui.draw_board(self.state)

    def step(self, ij) -> Tuple[np.ndarray, float, bool, None]:
        i, j = ij

        # record current status in history
        self.states.append(self.state.copy())
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
        self.state = self._state_orig.copy()
        self.update()

