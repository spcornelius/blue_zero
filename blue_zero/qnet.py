from typing import List, Union, Tuple

import torch
from torch.nn import Conv2d, Linear, Module, LeakyReLU, Sequential, \
    AdaptiveMaxPool2d, AdaptiveAvgPool2d
from blue_zero.config import Status
import blue_zero.util as util

__all__ = ['QNet']


class QNet(Module):

    def __init__(self, num_feat: int, num_hidden: int, depth: int,
                 kernel_size: Union[int, Tuple[int, int]] = (3, 3)):
        super().__init__()
        self.num_feat = num_feat
        self.num_hidden = num_hidden

        # pad with enough zeros so that convolutions preserve board shape
        try:
            h, w = kernel_size
            padding = (h - 2, w - 2)
        except TypeError:
            padding = kernel_size - 2

        # initial convolution (1 channel input --> num_feat channel output)
        layers = [Conv2d(4, num_feat,
                         kernel_size=kernel_size,
                         padding=padding,
                         bias=False),
                  LeakyReLU()]

        # subsequent convolutions (always with num_feat channels)
        for _ in range(depth - 1):
            layers.append(Conv2d(num_feat, num_feat,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 bias=False))
            layers.append(LeakyReLU())

        self.value = Sequential(Linear(3*num_feat, num_hidden),
                                LeakyReLU(),
                                Linear(num_hidden, 1)
                                )

        self.advantage = Sequential(Linear(4 * num_feat, num_hidden),
                                    LeakyReLU(),
                                    Linear(num_hidden, 1))

        self.embed = Sequential(*layers)

        # pool layers for calculate representations of board as a whole
        self.max_pool = AdaptiveMaxPool2d(output_size=(1, 1))
        self.avg_pool = AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, s: torch.Tensor,
                a: Union[torch.Tensor,
                         tuple, List[tuple]] = None) -> torch.Tensor:
        board_size = s.shape[-2:]
        h, w = board_size

        # treat like a batched state of shape (batch_size, h, w)
        # even if we only have one state
        s = s.view(-1, h, w)
        batch_size = len(s)

        # get bitboard representation
        s = util.to_bitboard(s)

        # calculating embedding by performing convolutions
        # e will have shape (batch_size, num_feat, h, w)
        a_rep = self.embed(s)

        # use pooling to calculate 3 different representations of board
        # as a whole.
        avg = self.avg_pool(a_rep)
        max_ = self.max_pool(a_rep)
        sum_ = h * w * avg
        s_rep = torch.cat((avg, max_, sum_), dim=1)

        # input_ will have shape (batch_size, 4 * num_feat, h, w)
        s_a_rep = torch.cat((s_rep.expand(-1, -1, h, w), a_rep), dim=1)

        # fully-connected Linear layers expect input dim (channels) to be
        # last, so permute dimensions to shape (batch_size, h, w, num_feat)
        s_a_rep = s_a_rep.permute(0, 2, 3, 1)
        s_rep = s_rep.permute(0, 2, 3, 1)

        adv = self.advantage(s_a_rep).view(batch_size, h, w)
        val = self.value(s_rep).expand(-1, 1, h, w).squeeze()

        invalid_mask = (s[:, Status.wall, :, :].byte() |
                        s[:, Status.attacked, :, :].byte()).bool()

        adv[invalid_mask] = 0

        num_actions = (adv != 0).sum(dim=(1, 2), keepdim=True).expand(-1, h, w)

        # finally, get q values for each square
        # q will have shape (batch_size, h, w)
        # q = self.w_final(hidden).view(batch_size, h, w)
        adv_mean = adv.sum(dim=(1, 2),
                           keepdim=True).expand(-1, h, w)/num_actions
        q = val + adv - adv_mean
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
