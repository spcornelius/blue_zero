from blue_zero.env import Blue
import blue_zero.config as cfg
import torch

__all__ = ['gen_random_env', 'init_weights', 'to_bitboard']


def gen_random_env(n, p, device='cpu'):
    return Blue.from_random((n, n), p, with_gui=False).to(device=device)


def init_weights(module, w_scale):
    def _init_weights(m):
        try:
            module.weight.data.uniform_(-w_scale, w_scale)
        except AttributeError:
            pass

        try:
            module.bias.data.uniform_(-w_scale, w_scale)
        except AttributeError:
            pass

    module.apply(_init_weights)


def to_bitboard(s: torch.Tensor):
    """ Turn a 2D (single board) or 3D (batch of boards) tensors whose
        elements are integer Status values to an equivalent bitboard.
        The return shape will be (batch_size, 4, h, w) if input was
        batched, otherwise (4, h, w) if unbatched, where h and w
        are the board height/width respectively. """
    dim = 0 if s.ndim == 2 else 1
    return torch.stack([s == status for status in cfg.Status],
                       dim=dim).to(dtype=torch.float)
