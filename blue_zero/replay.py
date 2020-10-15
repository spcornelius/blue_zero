import random
from collections import namedtuple, deque
from typing import Tuple

import torch

from blue_zero.env.blue import Blue

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

    def store(self, e: Blue) -> None:
        """ Memorize all n-step transitions in a terminal environment.

        Args:
            e: A terminal environment.
        """
        assert e.done
        for i in range(0, e.steps_taken):
            i_next = i + self.step_size
            a = e.actions[i]
            r_prev = e.rewards[i]
            s_prev = e.states[i]
            if i_next >= e.steps_taken:
                # final state
                s, r, terminal = e.state.clone(), e.r, True
            else:
                # non-final state
                s, r, terminal = e.states[i_next], e.rewards[i_next], False
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

        a = torch.stack(a, 0).to(device=device)
        dr = torch.tensor(dr).to(device=device)
        terminal = torch.tensor(terminal).to(device=device, dtype=torch.bool)
        s_prev = torch.stack(s_prev, 0).to(device=device)
        s = torch.stack(s, 0).to(device=device)

        return s_prev, a, s, dr, terminal

    def __len__(self):
        return len(self._memory)
