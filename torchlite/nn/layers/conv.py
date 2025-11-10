# torchlite/nn/layers/linear.py

from torchlite.nn.module import Module
from torchlite.autograd import Tensor
import numpy as np


class Conv2D(Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias

        # kaiming initialization
        limit = np.sqrt(1 / in_channels)

        w = np.random.uniform(-limit, limit, size=(out_channels, in_channels))
        self.weight = Tensor(w, requires_grad=True)

        if bias:
            b = np.random.uniform(-limit, limit, size=(out_channels,))
            self.bias = Tensor(b, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
      output = x @ self.weight.transpose()

    def __repr__(self):
        return f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, padding={self.padding}, stride={self.stride}, bias={self.bias is not None})"
