from dataclasses import dataclass
from functools import partial
from typing import Union, Tuple

import torch
from torch.nn import Conv2d, ReLU, Sequential

from blue_zero.config import Status
from blue_zero.qnet.base import QNet

__all__ = ['DuelingQNet']


@dataclass(eq=False)
class DuelingQNet(QNet, nickname='dueling'):
    num_feat: int
    num_hidden: int
    depth: int
    kernel_size: Union[int, Tuple[int, int]] = (3, 3)
    bias: bool = False

    def __post_init__(self):
        super().__post_init__()
        # pad with enough zeros so that convolutions preserve board shape
        try:
            h, w = self.kernel_size
            if (h % 2 == 0) or (w % 2 == 0):
                raise ValueError("all dimensions of kernel_size must be odd")
            padding = (h // 2, w // 2)
        except TypeError:
            if self.kernel_size % 2 == 0:
                raise ValueError("kernel_size must be odd")
            padding = self.kernel_size // 2

        # convolutions for embedding
        EmbedConv = partial(Conv2d,
                            kernel_size=self.kernel_size,
                            padding=padding,
                            bias=self.bias)

        # convolutions serving as fc layers (mixing channels post-embedding)
        FCConv = partial(Conv2d, kernel_size=(1, 1), bias=True)

        # initial convolution (1 channel input --> num_feat channel output)
        layers = [EmbedConv(4, self.num_feat), ReLU()]

        # subsequent convolutions (always with num_feat channels)
        for _ in range(self.depth - 1):
            layers.extend([EmbedConv(self.num_feat, self.num_feat),
                           ReLU()])

        self.embed = Sequential(*layers)

        self.value = Sequential(FCConv(3 * self.num_feat, self.num_hidden),
                                ReLU(),
                                FCConv(self.num_hidden, 1)
                                )

        self.advantage = Sequential(FCConv(4 * self.num_feat, self.num_hidden),
                                    ReLU(),
                                    FCConv(self.num_hidden, 1)
                                    )

    def q(self, s: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = s.shape

        # calculating embedding by performing convolutions
        # e will have shape (batch_size, num_feat, h, w)
        a_rep = self.embed(s)

        # use pooling to calculate 3 different representations of board
        # as a whole.
        avg = torch.mean(a_rep, dim=(2, 3), keepdim=True)
        max_ = torch.amax(a_rep, dim=(2, 3), keepdim=True)
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
        # invalid = (s[:, Status.wall, :, :].byte() |
        #           s[:, Status.attacked, :, :].byte()).bool()
        invalid = self.invalid_mask(s)
        adv.masked_fill_(invalid, 0.0)
        num_valid = (~invalid).sum(dim=(1, 2), keepdim=True)
        adv_mean = adv.sum(dim=(1, 2), keepdim=True) / num_valid

        # finally, get q values for each square
        # q will have shape (batch_size, h, w)
        return val + adv - adv_mean.expand_as(adv)
