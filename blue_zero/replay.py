import random
from collections import namedtuple, deque
from typing import Tuple

import torch

from blue_zero.env.base import BlueBase

__all__ = []
__all__.extend([
    'Transition',
    'NStepReplayMemory',
])

Transition = namedtuple('Transition', ('s_prev', 'a', 's', 'dr', 'terminal'))


class NStepReplayMemory(object):
    """ Replay buffer that memorizes n-step transitions an a reinforcement
    learning environment. """

    def __init__(self, capacity: int, step_size: int, device: str = 'cpu'):
        """
        Args:
            capacity: Maximum number of `Transition` objects to store in
                replay buffer before overwriting the earliest.
            step_size: Positive integer giving the 'N' in 'NStep'.
            device: Device to return samples on.
        """
        self.capacity = capacity
        self.step_size = step_size
        self.device = device
        self._memory = deque(maxlen=capacity)

    def store(self, e: BlueBase) -> None:
        """ Memorize all n-step transitions in a terminal environment.

        Args:
            e: A terminal environment.
        """
        assert e.done
        if e.steps_taken == 0:
            return
        for i in range(0, e.steps_taken):
            i_next = i + self.step_size
            a = torch.from_numpy(e.actions[i]).to(device=self.device)
            r_prev = e.rewards[i]
            s_prev = torch.from_numpy(e.states[i]).to(
                device=self.device).float()
            if i_next >= e.steps_taken:
                # final state
                s, r, terminal = e.state.copy(), e.r, True
            else:
                # non-final state
                s, r, terminal = e.states[i_next], e.rewards[i_next], False
            s = torch.from_numpy(s).to(device=self.device).float()
            self._memory.append(Transition(s_prev, a, s, r - r_prev, terminal))

    def sample(self, batch_size: int) -> Tuple:
        """
        Args:
            batch_size: Number of transitions to randomly sample and batch
                        together.

        Returns:
            A five-tuple (`s_prev`, `a`, `s`, `dr`, `terminal`), where `s`
            (`s_prev`) is the final (initial) state, `a` is the action taken
            in `s`, `dr` is the difference in cumulative reward between the
            final/initial states, and `terminal` is whether `s` is a terminal
            state.
        """
        device = self.device
        s_prev, a, s, dr, terminal = list(
            zip(*random.sample(self._memory, batch_size)))

        a = torch.stack(a)
        dr = torch.tensor(dr, device=device, dtype=torch.float32)
        terminal = torch.tensor(terminal, device=device, dtype=torch.bool)
        s_prev = torch.stack(s_prev)
        s = torch.stack(s)

        return s_prev, a, s, dr, terminal

    def __len__(self):
        return len(self._memory)
