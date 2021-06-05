from typing import Union, Tuple
from dataclasses import dataclass
import torch
from torch.nn import Conv2d, Module, ReLU, Sequential

from blue_zero.qnet.base import QNet

__all__ = ['DilationQNet']


class DilationEmbeddingBlock(Module):

    def __init__(self, c: int,
                 dilation: Union[int, Tuple[int, int]] = (1, 1),
                 kernel_size: Union[int, Tuple[int, int]] = (3, 3),
                 bias: bool = False):
        super().__init__()

        padding1 = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        padding2 = (dilation[0] * (kernel_size[0] - 1) // 2,
                    dilation[1] * (kernel_size[1] - 1) // 2)

        self.conv1 = Conv2d(c, c, kernel_size=kernel_size,
                            padding=padding1, dilation=1, bias=bias)
        self.conv2 = Conv2d(c, c, kernel_size=kernel_size,
                            padding=padding2, dilation=dilation, bias=bias)

        self.conv = Sequential(ReLU(), self.conv1, ReLU(), self.conv2)

    def forward(self, x: torch.Tensor):
        return self.conv(x) + x


@dataclass(eq=False)
class DilationQNet(QNet, id='dilation'):
    num_feat: int
    num_hidden: int
    depth: int
    dilation: Union[int, Tuple[int, int]] = (1, 1)
    kernel_size: Union[int, Tuple[int, int]] = (3, 3)
    bias: bool = False

    def __post_init__(self):
        super().__post_init__()

        try:
            self.dilation = tuple(self.dilation)
        except TypeError:
            self.dilation = (self.dilation, self.dilation)

        try:
            self.kernel_size = tuple(self.kernel_size)
        except TypeError:
            self.kernel_size = (self.kernel_size, self.kernel_size)

        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError("kernel_size must be odd")

        # initial convolution
        padding = ((self.kernel_size[0] - 1) // 2,
                   (self.kernel_size[1] - 1) // 2)
        embed_layers = [Conv2d(4, self.num_feat,
                               kernel_size=self.kernel_size,
                               bias=self.bias, padding=padding)]

        # subsequent blocks
        embed_layers.extend(
            DilationEmbeddingBlock(self.num_feat, self.dilation,
                                   self.kernel_size, bias=self.bias)
            for _ in range(self.depth)
        )
        self.embed = Sequential(*embed_layers)

        self.c1 = Conv2d(self.num_feat, self.num_hidden,
                         kernel_size=1, bias=True)
        self.c2 = Conv2d(self.num_hidden, 1,
                         kernel_size=1, bias=True)
        self._q = Sequential(ReLU(),
                             self.c1,
                             ReLU(),
                             self.c2
                             )

    def q(self, s: torch.Tensor) -> torch.Tensor:
        return self._q(self.embed(s))
