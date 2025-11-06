# torchlite/autograd/tensor.py
import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        self.data = data
        self.requires_grad = requires_grad

        self.grad = None  # holds the gradient. create on demand to save memory
        self._ctx = None  # ('op', Context)

    def backward(self, grad=None):
        from . import engine

        engine.backward(self, grad)

    def __add__(self, other):
        from . import ops

        other = other if isinstance(other, Tensor) else Tensor(other)
        return ops.add(self, other)

    def __mul__(self, other):
        from . import ops

        other = other if isinstance(other, Tensor) else Tensor(other)
        return ops.mul(self, other)

    def __pow__(self, other):
        from . import ops

        other = other if isinstance(other, Tensor) else Tensor(other)
        return ops.pow(self, other)

    def __sub__(self, other):
        from . import ops

        other = other if isinstance(other, Tensor) else Tensor(other)
        return ops.sub(self, other)

    def __neg__(self):
        from . import ops

        return ops.neg(self)

    def __matmul__(self, other):
        from . import ops

        other = other if isinstance(other, Tensor) else Tensor(other)
        return ops.matmul(self, other)

    def __rsub__(self, other):  # other - self
        from . import ops

        other = other if isinstance(other, Tensor) else Tensor(other)
        return ops.sub(other, self)

    __radd__ = __add__
    __rmul__ = __mul__

    def transpose(self, axes=None):
        from . import ops

        return ops.transpose(self, axes)

    def relu(self):
        from . import ops

        return ops.relu(self)

    def zero_grad(self):
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    def __repr__(self):
        return f"Tensor(data={self.data}), grad={self.grad})"

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape
