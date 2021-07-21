from typing import Union, Iterable

import numpy as np
import torch
import abc
from torch.nn.functional import softmax
from more_itertools import chunked
from tqdm import tqdm

from blue_zero.mode import BlueMode
from blue_zero.qnet import QNet

__all__ = []
__all__.extend([
    'Agent', 'QAgent', 'EpsGreedyQAgent', 'SoftMaxQAgent'
])


def flat_to_2d(idx, w):
    # convert 1d indices in a flattened row major grid to 2d
    return torch.cat(
        ((idx // w).view(-1, 1),
         (idx % w).view(-1, 1)), dim=1)


class Agent(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_action(self, s: torch.Tensor):
        pass


class QAgent(Agent):

    def __init__(self, net: QNet):
        self.net = net

    def _get_action(self, q: torch.Tensor):
        batch_size, h, w = q.shape
        q_new, ij_flat = torch.max(q.view(batch_size, -1), 1)
        ij = flat_to_2d(ij_flat, w)
        return ij, q_new

    def get_action(self, s: torch.Tensor,
                   return_q: bool = False):
        batched = s.ndim == 4

        with torch.no_grad():
            q = self.net(s)

        a, q = self._get_action(q)

        if not batched:
            a = a[0]
            q = q.squeeze()

        return (a, q) if return_q else a

    def play(self, envs: Iterable[BlueMode],
             batch_size: int = None,
             pbar: Union[bool, tqdm] = False) -> None:

        envs = list(envs)
        if batch_size is None:
            batch_size = len(envs)

        disable = pbar is False
        if pbar is True:
            pbar = tqdm(total=len(envs), desc="Playing")

        pbar.disable = disable
        pbar.update(sum(e.done for e in envs))

        for env_batch in chunked(envs, batch_size):
            while unfinished_envs := [e for e in env_batch if not e.done]:
                # batch the states together
                batch = np.stack([e.state for e in unfinished_envs])
                batch = torch.from_numpy(batch).to(device=self.net.device,
                                                   dtype=torch.float32)
                actions = self.get_action(batch).cpu().numpy()
                for env, a in zip(unfinished_envs, actions):
                    _, _, done, _ = env.step(a)
                    pbar.update(done)
                    pbar.refresh()


class EpsGreedyQAgent(QAgent):

    def __init__(self, net: QNet, eps: float = 0.0):
        super().__init__(net)
        self.eps = eps

    def _get_action(self, q: torch.Tensor):
        batch_size, h, w = q.shape

        # which boards in the batch should receive a random action?
        random = torch.rand(batch_size, device=q.device).lt(self.eps)
        random = random.view(-1, 1, 1).expand_as(q)

        # remember: ineligible squares have q = -infinity
        finite = torch.isfinite(q)

        # replace scores of eligible actions on designated boards with randoms
        # = sneaky way of getting random actions without writing a separate
        # function from the pure-greedy case
        q[finite & random] = torch.rand_like(q[finite & random])

        q_new, ij_flat = torch.max(q.view(batch_size, -1), 1)
        ij = flat_to_2d(ij_flat, w)
        return ij, q_new


class SoftMaxQAgent(QAgent):
    def __init__(self, net: QNet, T: float = 0.0):
        super().__init__(net)
        self.T = T

    def _get_action(self, q: torch.Tensor):
        batch_size, h, w = q.shape
        p = softmax(q.view(batch_size, -1) / self.T, dim=1).cpu().numpy()

        def choice(p_i):
            return np.random.choice(h * w, p=p_i.flatten())

        ij_flat = torch.tensor(list(map(choice, p)))
        ij = flat_to_2d(ij_flat, w)
        i, j = ij.t()
        q_new = q[torch.arange(batch_size), i, j]
        return ij, q_new
