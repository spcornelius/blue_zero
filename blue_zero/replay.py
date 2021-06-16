import random
from collections import namedtuple
from typing import Tuple

import torch
import numpy as np

from blue_zero.env.base import BlueEnv

__all__ = []
__all__.extend([
    'Transition',
    'NStepReplayMemory',
])

Transition = namedtuple('Transition',
                        ('s_prev', 'a', 's', 'dr', 'terminal'))


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

    def store(self, env: BlueEnv, gamma: float = 1.0) -> None:
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

        disc_rewards = [r * gamma ** k for k, r in enumerate(env.rewards)]
        cum_rewards = np.cumsum([0.0] + disc_rewards)

        for i_prev in range(0, env.steps_taken):
            i = i_prev + self.step_diff
            a = actions[i_prev]
            r_prev = cum_rewards[i_prev]
            s_prev = states[i_prev]

            if i >= env.steps_taken:
                # final state
                s = states[-1]
                r = cum_rewards[env.steps_taken]
                terminal = True
            else:
                # non-final state
                s = states[i]
                r = cum_rewards[i]
                terminal = False

            t = Transition(s_prev, a, s, r - r_prev, terminal)
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
        s_prev, a, s, dr, terminal = list(zip(*random.sample(self.buffer, n)))

        s_prev = torch.stack(s_prev)
        a = torch.stack(a)
        s = torch.stack(s)
        dr = torch.tensor(dr, device=self.device, dtype=torch.float32)
        terminal = torch.tensor(terminal, device=self.device, dtype=torch.bool)

        return s_prev, a, s, dr, terminal
