from dataclasses import dataclass
from typing import Union, Tuple

import torch
from torch.nn import Conv2d, ModuleList, BatchNorm2d
from torch.nn.functional import relu, leaky_relu
from functools import partial

from blue_zero.qnet.base import QNet

__all__ = ['SimpleQNet']


@dataclass(eq=False)
class SimpleQNet(QNet, id='simple'):
    num_feat: int
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
        self.convs = ModuleList([Conv2d(self.num_feat, self.num_feat,
                                        kernel_size=self.kernel_size,
                                        padding=padding,
                                        bias=self.bias) for _ in
                                 range(self.depth)])

        self.embed_input = Conv2d(4, self.num_feat,
                                  kernel_size=(1, 1),
                                  bias=self.bias)

        self.theta3 = Conv2d(2*self.num_feat, 1,
                             kernel_size=(1, 1),
                             bias=self.bias)
        self.theta1 = Conv2d(3 * self.num_feat, self.num_feat,
                             kernel_size=(1, 1),
                             bias=self.bias)
        self.theta2 = Conv2d(self.num_feat, self.num_feat,
                             kernel_size=(1, 1),
                             bias=self.bias)

        # self.theta1 = Conv2d(self.num_feat, 1,
        #                      kernel_size=(1, 1),
        #                      bias=self.bias)
        # self.theta4 = Conv2d(self.num_feat, 1,
        #                      kernel_size=(1, 1),
        #                      bias=self.bias)

    def q(self, s: torch.Tensor):
        batch_size, _, h, w = s.shape

        # relu = partial(leaky_relu, negative_slope=0.1)

        x = relu(self.embed_input(s))
        #y = torch.amax(x, dim=(2, 3), keepdim=True).expand(-1, -1, h, w)
        #x = torch.cat((x, y), dim=1)

        for k in range(self.depth):
            # y = torch.amax(x, dim=(2, 3), keepdim=True).expand(-1, -1, h, w)
            # z = torch.cat((x, y), dim=1)
            # x = x + relu(self.convs[k](z))
            x = x + relu(self.convs[k](x))

        # representation of action (a_rep)
        a_rep = x

        # global (pooled) features, representing board state (s_rep)
        avg = torch.mean(a_rep, dim=(2, 3), keepdim=True)
        max_ = torch.amax(a_rep, dim=(2, 3), keepdim=True)
        sum_ = h * avg
        s_rep = torch.cat((avg, max_, sum_), dim=1)

        # q = self.theta4(relu(self.theta3(
        #     relu(torch.cat((self.theta1(s_rep).expand(-1, -1, h, w),
        #                     self.theta2(a_rep)), dim=1))
        # )))

        q = self.theta3(
            relu(torch.cat((self.theta1(s_rep).expand(-1, -1, h, w),
                            self.theta2(a_rep)), dim=1))
        )
        # q = self.theta1(x)

        return q.squeeze()
