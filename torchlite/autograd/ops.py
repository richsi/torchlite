# torchlite/autograd/ops.py
import numpy as np
from .tensor import Tensor


class Context:
    """Stores intermediate results for backward pass."""

    def __init__(self):
        self.saved_tensors = ()  # op, ctx
        self.axes = None

    def save_for_backward(self, *args):
        self.saved_tensors = args


def add(a, b):
    out = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)
    if out.requires_grad:
        ctx = Context()
        ctx.save_for_backward(a, b)
        out._ctx = ("add", ctx)
    return out


def mul(a, b):
    out = Tensor(a.data * b.data, requires_grad=a.requires_grad or b.requires_grad)
    if out.requires_grad:
        ctx = Context()
        ctx.save_for_backward(a, b)
        out._ctx = ("mul", ctx)
    return out


def pow(a, b):
    out = Tensor(a.data**b.data, requires_grad=a.requires_grad or b.requires_grad)
    if out.requires_grad:
        ctx = Context()
        ctx.save_for_backward(a, b)
        out._ctx = ("pow", ctx)
    return out


def sub(a, b):
    out = Tensor(a.data + (-b.data), requires_grad=a.requires_grad or b.requires_grad)
    if out.requires_grad:
        ctx = Context()
        ctx.save_for_backward(a, b)
        out._ctx = ("sub", ctx)
    return out


def neg(a):
    out = Tensor(-a.data, requires_grad=a.requires_grad)
    if out.requires_grad:
        ctx = Context()
        ctx.save_for_backward(a)
        out._ctx = ("neg", ctx)
    return out


def matmul(a, b):
    out = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)
    if out.requires_grad:
        ctx = Context()
        ctx.save_for_backward(a, b)
        out._ctx = ("matmul", ctx)
    return out


def transpose(a, axes):
    if axes is None:
        axes = tuple(range(a.data.ndim - 1, -1, -1))

    out_data = np.transpose(a.data, axes)
    out = Tensor(out_data, requires_grad=True)

    if out.requires_grad:
        ctx = Context()
        ctx.save_for_backward(a)
        ctx.axes = axes
        out._ctx = ("transpose", ctx)

    return out


def relu(a):
    out_data = np.maximum(0, a.data)
    out = Tensor(out_data, requires_grad=a.requires_grad)
    if out.requires_grad:
        ctx = Context()
        ctx.save_for_backward(a)
        out._ctx = ("relu", ctx)
    return out


def mean(a):
    out_data = np.mean(a.data)
    out = Tensor(out_data, requires_grad=a.requires_grad)
    if out.requires_grad:
        ctx = Context()
        ctx.save_for_backward(a)
        out._ctx = ("mean", ctx)
    return out