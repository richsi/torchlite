# torchlite/autograd/engine.py
import numpy as np


def backward(tensor, grad=None):
    """
    Performs a backward pass starting from 'tensor'
    """

    # build the topological sort graph
    topo = []
    visited = set()

    def build_topo(t):
        if t not in visited:
            visited.add(t)
            if t.requires_grad and t._ctx:
                _, ctx = t._ctx
                for input_tensor in ctx.saved_tensors:
                    build_topo(input_tensor)
            topo.append(t)

    build_topo(tensor)

    # all .grad attributes should be zero arrays before accumulating grads
    for t in topo:
        if t.requires_grad:
            if t.grad is None:
                t.grad = np.zeros_like(t.data, dtype=np.float32)

    # starting gradient should be 1
    if grad is None:
        if tensor.data.size != 1:
            raise RuntimeError("grad must be specified for non-scalar Tensors")
        tensor.grad = np.ones_like(tensor.data, dtype=np.float32)
    else:
        # ensure grad is np array with correct shape
        grad_np = np.asarray(grad, dtype=np.float32)
        if grad_np.shape != tensor.shape:
            raise ValueError(
                f"Gradient shape {grad_np.shape} must match tensor shape {tensor.shape}"
            )
        tensor.grad = grad_np

    # apply chain rule
    for t in reversed(topo):
        if t._ctx:
            op, ctx = t._ctx
            up_grad = t.grad

            # calling backwards based on op
            backward_fn = op_map[op]
            backward_fn(ctx, up_grad)


def _handle_broadcast(grad, tensor_shape):
    """
    Handles broadcasting in gradients. The gradient 'grad' may have
    a large shape than 'tensor_shape' due to broadcasting rules.
    Sum gradient along broadcasted axes.
    """

    if grad.shape == tensor_shape:
        return grad

    # scalar expansion (scalar tensor, vector grad)
    if tensor_shape == ():
        return np.sum(grad)

    # prepended dimensions e.g., grad.shape = (5, 3, 4), tensor_shape = (3, 4)
    n_extra_dims = grad.ndim - len(tensor_shape)
    if n_extra_dims > 0:
        grad = np.sum(
            grad, axis=tuple(range(n_extra_dims))
        )  # sum and *remove* these dims

    # handle stretched dimensions (1's) e.g., grad.shape = (3, 5), tensor_shape = (3, 1)
    axes_to_sum = []
    for i, dim in enumerate(tensor_shape):
        if dim == 1 and grad.shape[i] > 1:
            axes_to_sum.append(i)

    if axes_to_sum:
        # keep dims true so that (5, 1) doesnt become (5,)
        grad = np.sum(grad, axis=tuple(axes_to_sum), keepdims=True)

    assert (
        grad.shape == tensor_shape
    ), f"grad.shape {grad.shape} does not match tensor_shape {tensor_shape}"

    return grad


def add_backward(ctx, grad):
    # a + b
    a, b = ctx.saved_tensors
    if a.requires_grad:  # chain rule: grad * 1
        a.grad += _handle_broadcast(grad, a.shape)
    if b.requires_grad:
        b.grad += _handle_broadcast(grad, b.shape)


def mul_backward(ctx, grad):
    # a * b
    a, b = ctx.saved_tensors
    if a.requires_grad:
        a.grad += _handle_broadcast(grad * b.data, a.shape)
    if b.requires_grad:
        b.grad += _handle_broadcast(grad * a.data, b.shape)


def pow_backward(ctx, grad):
    # a ** b
    a, b = ctx.saved_tensors
    if a.requires_grad:
        # Local gradient is b * a**(b-1)
        grad_a = grad * (b.data * a.data ** (b.data - 1))
        a.grad += _handle_broadcast(grad_a, a.shape)
    if b.requires_grad:
        # Local gradient is (a**b) * log(a)
        epsilon = 1e-9  # small episolon to avoid log(0)
        grad_b = grad * ((a.data**b.data) * np.log(a.data + epsilon))
        b.grad += _handle_broadcast(grad_b, b.shape)


def sub_backward(ctx, grad):
    # a - b
    a, b = ctx.saved_tensors
    if a.requires_grad:
        # Local gradient is 1
        a.grad += _handle_broadcast(grad, a.shape)
    if b.requires_grad:
        # Local gradient is -1
        b.grad += _handle_broadcast(-grad, b.shape)


def neg_backward(ctx, grad):
    # -a
    t, = ctx.saved_tensors
    if t.requires_grad:
        t.grad += _handle_broadcast(-grad, t.shape)


def matmul_backward(ctx, grad):
    a, b = ctx.saved_tensors
    if a.requires_grad:
        # grad_A = grad_C @ B.T
        a.grad += _handle_broadcast(grad @ b.data.T, a.shape)
    if b.requires_grad:
        # grad_B = A.T @ grad_C
        b.grad += _handle_broadcast(a.data.T @ grad, b.shape)


def transpose_backward(ctx, grad):
    a, axes = ctx.saved_tensors
    if a.requires_grad:
        backward_axes = np.argsort(axes)
        inverse_perm = np.transpose(grad, backward_axes)  # undoes the initial transpose
        a.grad += _handle_broadcast(inverse_perm, a.shape)


def relu_backward(ctx, grad):
    t, = ctx.saved_tensors
    if t.requires_grad:
        # 1 if t.data > 0 else 0
        local_grad = (t.data > 0).astype(t.data.dtype)
        t.grad += _handle_broadcast(grad * local_grad, t.shape)

def mean_backward(ctx, grad):
    t, = ctx.saved_tensors
    if t.requires_grad:
        N = t.data.size
        local_grad_scalar = grad / N
        grad_out = np.full_like(t.data, fill_value=local_grad_scalar)
        t.grad += grad_out # no handle_broadcast since shape is matched


op_map = {
    "add": add_backward,
    "mul": mul_backward,
    "pow": pow_backward,
    "sub": sub_backward,
    "neg": neg_backward,
    "matmul": matmul_backward,
    "transpose": transpose_backward,
    "relu": relu_backward,
    "mean": mean_backward,
}
