import blue_zero.config as cfg
import torch
import os
import random
import numpy as np
from wurlitzer import pipes
from torch.nn import Conv2d


__all__ = ['init_weights', 'to_bitboard', 'set_seed']


def init_weights(module):
    def _init_weights(m):
        if isinstance(m, Conv2d):
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        try:
            torch.nn.init.zeros_(m.bias)
        except AttributeError:
            pass

    module.apply(_init_weights)


def to_bitboard(s: np.ndarray):
    """ Turn a 2D (single board) or 3D (batch of boards) tensors whose
        elements are integer Status values to an equivalent bitboard.
        The return shape will be (batch_size, 4, h, w) if input was
        batched, otherwise (4, h, w) if unbatched, where h and w
        are the board height/width respectively. """
    axis = 0 if s.ndim == 2 else 1
    layers = [s == status for status in cfg.Status]
    return np.stack(layers, axis=axis)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    with pipes() as (_, _):
        torch.set_deterministic(True)
