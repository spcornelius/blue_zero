from typing import List, Union, Tuple

import torch
from torch.nn import Conv2d, Module, ReLU, Sequential, \
    AdaptiveMaxPool2d, AdaptiveAvgPool2d

import blue_zero.util as util
from blue_zero.config import Status

__all__ = ['DQN']


class DQN(Module):

    def __init__(self, num_feat: int, num_hidden: int, depth: int,
                 kernel_size: Union[int, Tuple[int, int]] = (3, 3),
                 with_conv_bias: bool = False):
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
                         bias=with_conv_bias),
                  ReLU()]

        # subsequent convolutions (always with num_feat channels)
        for _ in range(depth - 1):
            layers.append(Conv2d(num_feat, num_feat,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 bias=with_conv_bias))
            layers.append(ReLU())

        self.embed = Sequential(*layers)

        self.value = Sequential(Conv2d(3 * num_feat, num_hidden,
                                       kernel_size=(1, 1), bias=True),
                                ReLU(),
                                Conv2d(num_hidden, 1,
                                       kernel_size=(1, 1), bias=True)
                                )

        self.advantage = Sequential(Conv2d(4 * num_feat, num_hidden,
                                           kernel_size=(1, 1), bias=True),
                                    ReLU(),
                                    Conv2d(num_hidden, 1,
                                           kernel_size=(1, 1), bias=True))

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

        # representation of board state as a whole
        # shape is (batch_size, 3 * num_feat, 1)
        s_rep = torch.cat((avg, max_, sum_), dim=1)

        # board representation + action representations combined
        # board state is simply repeated for every square in that board
        # s_a_rep will thus have shape (batch_size, 4 * num_feat, h, w)
        s_a_rep = torch.cat((s_rep.expand(-1, -1, h, w), a_rep), dim=1)

        # calculate advantage for each action (h x w) and value for each
        # board state as a whole. Represent both as shape
        # (batch_size x h x w)
        adv = self.advantage(s_a_rep).view(batch_size, h, w)
        val = self.value(s_rep).expand(-1, 1, h, w).squeeze()

        # only want to calculate the average advantage over *valid* actions
        # zero out that of ineligible moves
        invalid = (s[:, Status.wall, :, :].byte() |
                   s[:, Status.attacked, :, :].byte()).bool()
        adv.masked_fill_(invalid, 0.0)
        adv_mean = adv.sum(dim=(1, 2), keepdim=True) / \
            (~invalid).sum(dim=(1, 2), keepdim=True)

        # finally, get q values for each square
        # q will have shape (batch_size, h, w)
        # q = self.w_final(hidden).view(batch_size, h, w)
        q = val + adv - adv_mean.expand_as(adv)

        # give ineligible moves a score of -infinity
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
