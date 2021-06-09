import random
from collections import namedtuple, deque
from typing import Tuple

import torch

from blue_zero.env.base import BlueEnv

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

    def __init__(self, capacity: int, batch_size: int, step_diff: int = 1):
        """
        Args:
            capacity: Maximum number of `Transition` objects to store in
                replay buffer before overwriting the earliest.
            batch_size: Size of minibatch.
            step_diff: Positive integer corresponding to the 'N' in 'NStep'.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.step_diff = step_diff
        self._memory = deque(maxlen=capacity)

    def store(self, env: BlueEnv) -> None:
        """ Memorize all n-step transitions in a terminal environment.

        Args:
            env: A terminal environment.
        """
        assert env.done
        if env.steps_taken == 0:
            return

        # convert a numpy array in the environment to a torch tensor
        # for storage in replay buffer
        def torchify(x, dtype):
            return torch.from_numpy(x).to(device='cpu', dtype=dtype)

        # Convert environment states (numpy arrays) to tensors all at once.
        # If state appears multiple times in replay, should be stored as
        # references rather than multiple copies.
        states = [torchify(s, torch.float32) for s in env.states]
        states.append(torchify(env.state, torch.float32))
        actions = [torchify(a, torch.long) for a in env.actions]

        for i_prev in range(0, env.steps_taken):
            i = i_prev + self.step_diff
            a = actions[i_prev]
            r_prev = env.rewards[i_prev]
            s_prev = states[i_prev]

            if i >= env.steps_taken:
                # final state
                s = states[-1]
                r = env.r
                terminal = True
                dt = self.step_diff
            else:
                # non-final state
                s = states[i]
                r = env.rewards[i]
                terminal = False
                dt = env.steps_taken - i

            self._memory.append(Transition(s_prev, a, s, r - r_prev, terminal,
                                           dt))

    def sample(self, device: str = 'cpu') -> Tuple:
        """
        Args:
            device: Device on which to put batched tensors.

        Returns:
            A five-tuple (`s_prev`, `a`, `s`, `dr`, `terminal`), where `s`
            (`s_prev`) is the final (initial) state, `a` is the action taken
            in `s`, `dr` is the difference in cumulative reward between the
            final/initial states, and `terminal` is whether `s` is a terminal
            state.
        """
        # device = self.device
        s_prev, a, s, dr, terminal, dt = list(
            zip(*random.sample(self._memory, self.batch_size)))

        s_prev = torch.stack(s_prev).to(device=device)
        a = torch.stack(a).to(device=device)
        s = torch.stack(s).to(device=device)
        dr = torch.tensor(dr, device=device, dtype=torch.float32)
        terminal = torch.tensor(terminal, device=device, dtype=torch.bool)
        dt = torch.tensor(dt, device=device, dtype=torch.float32)

        return s_prev, a, s, dr, terminal, dt

    def __len__(self):
        return len(self._memory)
