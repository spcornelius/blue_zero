import random
from collections import namedtuple
from typing import Tuple

import torch
import numpy as np

from blue_zero.mode.base import BlueMode

__all__ = []
__all__.extend([
    'Transition',
    'NStepReplayMemory',
])

Transition = namedtuple('Transition',
                        ('s_prev', 'a', 's', 'dr', 'terminal', 'dt'))


class NStepReplayMemory(object):
    """ Replay buffer that memorizes n-step transitions an a reinforcement
    learning environment. """

    def __init__(self, capacity: int, step_diff: int = 1,
                 device: str = 'cpu'):
        """
        Args:
            capacity: Maximum number of `Transition` objects to store in
                replay buffer before overwriting the earliest.
            step_diff: Positive integer corresponding to the 'N' in 'NStep'.
        """
        self.capacity = capacity
        self.step_diff = step_diff
        self.buffer = []
        self.pos = 0
        self.device = device

    def __len__(self):
        return len(self.buffer)

    def store(self, env: BlueMode, gamma: float = 1.0) -> None:
        """ Memorize all n-step transitions in a terminal environment.

        Args:
            env: A terminal environment.
            gamma: Discount factor for rewards.
        """
        assert env.done
        if env.steps_taken == 0:
            return

        # convert a numpy array in the environment to a torch tensor
        # for storage in replay buffer
        def torchify(x, dtype):
            return torch.from_numpy(x).to(device=self.device, dtype=dtype)

        # Convert environment states (numpy arrays) to tensors all at once.
        # If state appears multiple times in replay, should be stored as
        # references rather than multiple copies.
        states = [torchify(s, torch.float32) for s in env.states]
        states.append(torchify(env.state, torch.float32))
        actions = [torchify(a, torch.long) for a in env.actions]
        # cum_rewards = np.cumsum([0.0] + [r*gamma**k for k, r in enumerate(mode.rewards)])
        # cum_rewards = (cum_rewards - np.mean(cum_rewards)) / np.std(cum_rewards)

        for i_prev in range(0, env.steps_taken):
            a = actions[i_prev]
            s_prev = states[i_prev]

            i = min(i_prev + self.step_diff, env.steps_taken)
            s = states[i]
            r = sum(gamma**k * env.rewards[k] for k in range(i-i_prev))
            # r = cum_rewards[i] - cum_rewards[i_prev]
            terminal = i == env.steps_taken
            dt = i - i_prev

            t = Transition(s_prev, a, s, r, terminal, dt)
            if len(self) < self.capacity:
                self.buffer.append(t)
            else:
                self.buffer[self.pos] = t
                self.pos = (self.pos + 1) % self.capacity

    def sample(self, n: int) -> Tuple:
        """
        Args:
            n: Number of experiences to sample (uniformly, with replacement).

        Returns:
            A five-tuple (`s_prev`, `a`, `s`, `dr`, `terminal`), where `s`
            (`s_prev`) is the final (initial) state, `a` is the action taken
            in `s`, `dr` is the difference in cumulative reward between the
            final/initial states, and `terminal` is whether `s` is a terminal
            state.
        """
        s_prev, a, s, dr, terminal, dt = \
            list(zip(*random.choices(self.buffer, k=n)))

        s_prev = torch.stack(s_prev)
        a = torch.stack(a)
        s = torch.stack(s)
        dr = torch.tensor(dr, device=self.device, dtype=torch.float32)
        terminal = torch.tensor(terminal, device=self.device, dtype=torch.bool)
        dt = torch.tensor(dt, device=self.device, dtype=torch.float32)

        return s_prev, a, s, dr, terminal, dt
