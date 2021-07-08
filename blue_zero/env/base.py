from typing import Tuple, Union

import numpy as np
from gym import Env

import blue_zero.config as cfg
from blue_zero.config import Status
from blue_zero.gui import BlueGUI
import blue_zero.util as util

__all__ = []
__all__.extend([
    'BlueEnv', 'mode_registry'
])

mode_registry = {}


class BlueEnv(Env):

    def __init__(self, state: '2D array_like',
                 show_gui: bool = False,
                 screen_size: Tuple[int, int] = cfg.screen_size,
                 reward_norm: Union[str, float] = None,
                 shape_rewards: bool = False,
                 **kwargs):
        super().__init__()

        state = util.to_bitboard(np.asarray(state, dtype=np.float32))

        # sanity tests
        assert state.ndim == 3
        assert state.shape[0] == len(cfg.Status)
        assert np.all(np.sum(state, axis=0) == 1)

        self.state = state

        _, h, w = self.state.shape
        if reward_norm is None:
            self._reward_norm = 1.0
        elif reward_norm == 'board_size':
            self._reward_norm = h * w
        elif reward_norm == 'side_length':
            self._reward_norm = max(h, w)
        elif isinstance(reward_norm, float):
            self._reward_norm = reward_norm
        else:
            raise ValueError(
                f"Unrecognized reward normalization strategy: {reward_norm}")

        self.show_gui = show_gui
        self.shape_rewards = shape_rewards
        self._green_pct_init = self.state[Status.alive].mean()

        self.states = []
        self.actions = []
        self.rewards = []
        self.steps_taken = 0

        # keep copy of initial state for resetting
        self._state_orig = self.state.copy()
        self._game_over = False

        if self.show_gui:
            self.gui = BlueGUI(self.state.shape[1:],
                               screen_size=screen_size)

        self.update()

    # noinspection PyMethodOverriding
    # noinspection PyShadowingBuiltins
    def __init_subclass__(cls, id: str, **kwargs):
        if id in mode_registry:
            raise TypeError(
                f"There is already a subclass of BlueEnv registered with "
                f"id {id}.")
        cls._id = id
        mode_registry[id] = cls

    # noinspection PyShadowingBuiltins
    @staticmethod
    def create(id: str, board: np.ndarray, **kwargs):
        try:
            return mode_registry[id](board, **kwargs)
        except KeyError:
            raise ValueError(
                f"Can't find a subclass of BlueEnv with id '{id}'.")

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
        return int(np.sum(np.abs(self.rewards)) * self._reward_norm)

    # noinspection PyUnusedLocal
    def reward(self, action) -> float:
        return -1.0 / self._reward_norm

    @property
    def action_space(self) -> np.ndarray:
        return np.argwhere(self.state[Status.alive, :, :])

    def update(self) -> None:
        if self.show_gui:
            self.render()

    def render(self, mode='human'):
        self.gui.draw_board(self.state)

    def step(self, ij) -> Tuple[np.ndarray, float, bool, None]:
        i, j = ij

        # record current status in history
        self.states.append(self.state.copy())
        self.actions.append(ij)

        # get reward
        r = self.reward(ij)
        self.rewards.append(r)

        # update state (flip bit from alive to attacked
        assert self.state[Status.alive, i, j]
        self.state[Status.alive, i, j] = False
        self.state[Status.attacked, i, j] = True

        # recompute clusters and check termination
        self.update()

        # add adjust reward up for the fraction of green squares cleared
        if self.shape_rewards:
            green_pct_prev = self.states[-1][Status.alive].mean()
            green_pct = self.state[Status.alive].mean()

            r += (green_pct_prev - green_pct)/self._green_pct_init

        self.steps_taken += 1

        return self.state, r, self._game_over, None

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.steps_taken = 0
        self.state = self._state_orig.copy()
        self._game_over = False
        self.update()
