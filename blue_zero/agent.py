from __future__ import annotations

from copy import deepcopy
from time import sleep
from typing import Union, List

import numpy as np
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

from blue_zero.env import BlueBase
from blue_zero.qnet import QNet

__all__ = []
__all__.extend([
    'Agent'
])


def flat_to_2d(idx, w):
    # convert 1d indices in a flattened row major grid to 2d
    return torch.cat(
        ((idx // w).view(-1, 1),
         (idx % w).view(-1, 1)), dim=1)


def get_action_greedy_eps(q, eps):
    batch_size, h, w = q.shape

    # which boards in the batch should receive a random action?
    random = torch.rand_like(q).lt(eps).view(-1, 1, 1).expand_as(q)

    # remember: ineligible squares have q = -infinity
    finite = torch.isfinite(q)

    # replace scores of eligible actions on designated boards with randoms
    # = sneaky way of getting random actions without writing a separate
    # function from the pure-greedy case
    q[finite & random] = torch.rand_like(q[finite & random])

    q_new, ij_flat = torch.max(q.view(batch_size, -1), 1)
    ij = flat_to_2d(ij_flat, w)
    return ij, q_new


def get_action_softmax(q, temp):
    batch_size, h, w = q.shape
    p = softmax(q.view(batch_size, -1) / temp, dim=1).cpu().numpy()

    def choice(p_i):
        return np.random.choice(h * w, p=p_i.flatten())

    ij_flat = torch.tensor(list(map(choice, p)))
    ij = flat_to_2d(ij_flat, w)
    i, j = ij.t()
    q_new = q[torch.arange(batch_size), i, j]
    return ij, q_new


class Agent(object):

    def __init__(self, net: QNet):
        self.net = net
        self.net.train()

    def get_action(self, s: torch.Tensor,
                   eps: float = 0.0,
                   return_q: bool = False):
        batched = s.ndim == 3

        with torch.no_grad():
            q = self.net(s)

        a, q = get_action_greedy_eps(q, eps)

        if not batched:
            a = a[0]
            q = q.squeeze()

        return (a, q) if return_q else a

    def play_envs(self, envs: List[BlueBase],
                  eps: float = 0.0,
                  pbar: Union[bool, tqdm] = False,
                  pause: float = 0.,
                  device='cpu') -> None:

        envs = list(envs)
        close_when_done = False
        if pbar is True:
            pbar = tqdm(total=len(envs), desc="Playing")
            close_when_done = True

        pbar.update(sum(e.done for e in envs))
        while unfinished_envs := [e for e in envs if not e.done]:
            # batch the states together
            batch = np.stack([e.state for e in unfinished_envs])
            batch = torch.from_numpy(batch).to(device=device)
            actions = self.get_action(batch, eps=eps).cpu().numpy()
            for i, (e, a) in enumerate(zip(unfinished_envs, actions)):
                _, _, done, _ = e.step(a)

                if done and pbar:
                    pbar.update(1)
                    pbar.refresh()

            sleep(pause)

        if pbar and close_when_done:
            pbar.close()

    def copy(self) -> Agent:
        return self.__class__(deepcopy(self.net))

    def update(self, other):
        self.net.load_state_dict(other.net.state_dict())

    def parameters(self):
        return self.net.parameters()

    def train(self) -> None:
        self.net.train()

    def eval(self) -> None:
        self.net.eval()

    def to(self, *args, **kwargs) -> Agent:
        self.net.to(*args, **kwargs)
        return self
