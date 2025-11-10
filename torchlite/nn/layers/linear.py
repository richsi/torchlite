# torchlite/nn/layers/linear.py

from torchlite.nn.module import Module
from torchlite.autograd import Tensor
import numpy as np


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # kaiming initialization
        limit = np.sqrt(1 / in_features)

        # size (out_features, in_features) for performance - matrices stored in row-major order
        w = np.random.uniform(-limit, limit, size=(out_features, in_features))
        self.weight = Tensor(w, requires_grad=True)

        if bias:
            b = np.random.uniform(-limit, limit, size=(out_features,))
            self.bias = Tensor(b, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        # x shape (batch_size, in_features)

        # Y = XW.T + b
        output = x @ self.weight.transpose()
        if self.bias is not None:
            # engine will handle broadcasting (batch_size, out_features) + (out_features,)
            output = output + self.bias

        return output

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
