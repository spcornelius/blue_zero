import abc
from dataclasses import dataclass, is_dataclass, asdict
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from torch.nn import Module, Parameter

from blue_zero.config import Status

__all__ = ['QNet']

# map human-readable qnet identifiers ('dueling', etc.) to the appropriate
# subclass
_registry = {}


@dataclass(eq=False)
class QNet(Module, metaclass=abc.ABCMeta):

    def __post_init__(self):
        super().__init__()
        self.dummy_param = Parameter(torch.empty(0))

    # noinspection PyShadowingBuiltins
    @staticmethod
    def create(id: str, **kwargs):
        try:
            return _registry[id](**kwargs)
        except KeyError:
            raise ValueError(
                f"Can't find a subclass of QNet with id '{id}'.")

    # noinspection PyMethodOverriding
    # noinspection PyShadowingBuiltins
    def __init_subclass__(cls, id: str, **kwargs):
        if not is_dataclass(cls):
            raise TypeError(
                "Subclasses of QNet must be dataclasses.")

        if id in _registry:
            raise TypeError(
                f"There is already a subclass of QNet registered with"
                f"name '{id}'.")
        cls._id = id
        _registry[id] = cls

    def save(self, file: Union[str, Path]):
        qnet_args = asdict(self)
        qnet_args['id'] = self._id
        data = dict(state_dict={k: v.cpu() for k, v in
                                self.state_dict().items()},
                    qnet_args=qnet_args)
        torch.save(data, file)

    @staticmethod
    def load(file: Union[str, Path]):
        data = torch.load(file)
        net = QNet.create(**data['qnet_args'])
        net.load_state_dict(data['state_dict'])
        return net

    @abc.abstractmethod
    def q(self, s: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def invalid_mask(s: torch.Tensor):
        return ~s[:, Status.alive, :, :].bool()

    def forward(self, s: Union[torch.Tensor, np.ndarray],
                a: Union[torch.Tensor,
                         tuple, List[tuple]] = None) -> torch.Tensor:
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).to(dtype=torch.float32)
        board_size = s.shape[-2:]
        h, w = board_size

        # treat like a batched state of shape (batch_size, 4, h, w)
        # even if we only have one state
        s = s.view(-1, len(Status), h, w)
        batch_size = len(s)

        # get bitboard representation
        # s = util.to_bitboard(s)

        # only want to calculate the average advantage over *valid* actions
        # zero out that of ineligible moves
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

    @property
    def device(self):
        return self.dummy_param.device
