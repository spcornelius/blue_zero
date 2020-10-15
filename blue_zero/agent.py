from __future__ import annotations
from copy import deepcopy
from typing import Iterable

import numpy as np
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

from blue_zero.config import Status
from blue_zero.env.blue import Blue
from time import sleep

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
    random = (torch.rand(len(q),
                         device=q.device) < eps).view(-1, 1, 1).expand_as(q)

    # remember: ineligible squares have q = -infinity
    finite = ~torch.isinf(q)

    # sneaky way of getting random actions from teh same function used
    # to get "best" actions, depending on x
    q[finite & random] = torch.rand_like(q[finite & random])

    q_new, ij_flat = torch.max(q.view(batch_size, -1), 1)
    ij = flat_to_2d(ij_flat, w)
    assert (~torch.isinf(q_new)).all().item()
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
    assert (~torch.isinf(q_new)).all().item()
    return ij, q_new


class Agent(object):

    def __init__(self, net: torch.nn.Module):
        self.net = net
        self.net.train()

    def get_action(self, s: torch.Tensor,
                   eps: float = 0.0,
                   return_q: bool = False):
        input_was_batched = s.ndim == 3
        with torch.no_grad():
            q = self.net(s)

        # replace ineligible squares with -infinity
        # (tricky way of ensuring they are never chosen by a
        # maximization_based strategy)
        invalid = (s == Status.wall) | (s == Status.attacked)
        q[invalid.view_as(q)] = -float('inf')

        a, q = get_action_greedy_eps(q, eps)

        for b, ij in enumerate(a):
            i, j = tuple(ij)

        if not input_was_batched:
            a = a[0]
            q = q.squeeze().item()

        return (a, q) if return_q else a

    def play_envs(self, envs: Iterable[Blue],
                  eps: float = 0.0,
                  with_pbar: bool = False,
                  pause=0.0) -> None:

        envs = list(envs)
        if with_pbar:
            pbar = tqdm(total=len(envs))

        while unfinished_envs := [e for e in envs if not e.done]:
            # batch the states together
            batch = torch.stack([e.state for e in unfinished_envs])
            actions, q_vals = self.get_action(batch,
                                              eps=eps,
                                              return_q=True)

            for i, (e, a) in enumerate(zip(unfinished_envs, actions)):
                _, _, done, _ = e.step(a)

                if done and with_pbar:
                    pbar.update(1)

            sleep(pause)

        if with_pbar:
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
