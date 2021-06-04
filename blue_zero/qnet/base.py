import abc
from dataclasses import dataclass, is_dataclass, asdict
from typing import Union, List

import numpy as np
import torch
from torch.nn import Module
from path import Path

import blue_zero.util as util
from blue_zero.config import Status

__all__ = ['QNet']

# map qnet names ('dueling', etc.) to the appropriate subclass
_registry = {}


@dataclass(eq=False)
class QNet(Module, metaclass=abc.ABCMeta):

    def __post_init__(self):
        super().__init__()

    @staticmethod
    def create(type_: str, **kwargs):
        try:
            return _registry[type_](**kwargs)
        except KeyError:
            raise ValueError(f"Unrecognized Q network type '{type_}.")

    def __init_subclass__(cls, **kwargs):
        if not is_dataclass(cls):
            raise TypeError(
                "Subclasses of QNet must be dataclasses.")

        try:
            nickname = kwargs['nickname']
        except KeyError:
            raise TypeError(
                f"Subclass {cls.__name__} of QNet is missing required class "
                f"kwarg 'nickname'.")

        if nickname in _registry:
            raise TypeError(
                f"nickname '{nickname}' of subclass {cls.__name__} of QNet "
                f"conflicts with that of another.")
        cls._nickname = nickname
        _registry[nickname] = cls

    def save(self, file: Union[str, Path]):
        qnet_params = asdict(self)
        qnet_params['type'] = self._nickname
        state = dict(state_dict={k: v.cpu() for k, v in
                                 self.state_dict().items()},
                     qnet_params=qnet_params)
        torch.save(state, file)

    @staticmethod
    def load(file: Union[str, Path]):
        data = torch.load(file)
        qnet_params = data['qnet_params']
        qnet_type = data.pop('type')
        new_net = QNet.create(qnet_type, **qnet_params)
        new_net.load_state_dict(data['state_dict'])
        return new_net

    @abc.abstractmethod
    def q(self, s: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def invalid_mask(s: torch.Tensor):
        return ~s[:, Status.alive, :, :].bool()

    def forward(self, s: Union[torch.Tensor, np.ndarray],
                a: Union[torch.Tensor,
                         tuple, List[tuple]] = None) -> torch.Tensor:
        s = torch.as_tensor(s)
        board_size = s.shape[-2:]
        h, w = board_size

        # treat like a batched state of shape (batch_size, h, w)
        # even if we only have one state
        s = s.view(-1, h, w)
        batch_size = len(s)

        # get bitboard representation
        s = util.to_bitboard(s)

        # only want to calculate the average advantage over *valid* actions
        # zero out that of ineligible moves
        #invalid = (s[:, Status.wall, :, :].byte() |
        #           s[:, Status.attacked, :, :].byte()).bool()

        q = self.q(s).view(batch_size, h, w)
        invalid = self.invalid_mask(s)
        q.masked_fill_(invalid, float("-inf"))
        if a is None:
            # user wants all actions for all boards in batch
            return q
        else:
            # user wants the q value for one specific action for each board
            if batch_size == 1:
                return q[a]
            else:
                i, j = a.t()
                batch_idx = torch.arange(0, batch_size, device=a.device)
                return q[batch_idx, i, j]
