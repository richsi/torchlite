import numpy as np
from numpy.testing import assert_allclose

from torchlite.autograd.tensor import Tensor
from torchlite.autograd.engine import backward, _handle_broadcast


# --- Test _handle_braodcast ---
def test_broadcast_no_op():
    grad = np.ones((3, 3))
    shape = (3, 3)
    assert_allclose(_handle_broadcast(grad, shape), grad)


def test_broadcast_scalar_expansion():
    grad = np.ones((3, 3)) * 2
    shape = ()
    assert_allclose(_handle_broadcast(grad, shape), 18.0)


def test_broadcast_prepended_dims():
    grad = np.ones((5, 3, 4))
    shape = (3, 4)
    result = _handle_broadcast(grad, shape)
    assert result.shape == shape
    assert_allclose(result, np.full((3, 4), 5.0))


def test_broadcast_squeezed_dims():
    grad = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    shape = (1, 3)
    result = _handle_broadcast(grad, shape)
    assert result.shape == shape
    assert_allclose(result, np.array([[5, 7, 9]]))


def test_broadcast_complex():
    grad = np.ones((5, 3, 1, 4))
    shape = (3, 1, 1)
    # 1. Sum prepended dim (axis 0) -> (3, 1, 4)
    # 2. Sum broadcasted dim (axis 2) -> (3, 1, 1)
    result = _handle_broadcast(grad, shape)
    assert result.shape == shape
    assert_allclose(result, np.full((3, 1, 1), 20.0))


# --- Test backprop ---


def test_add_backward():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    c = a + b

    grad_c = np.array([10, 20, 30])
    backward(c, grad=grad_c)
    assert_allclose(a.grad, [10, 20, 30])
    assert_allclose(b.grad, [10, 20, 30])


def test_mul_backward():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    c = a * b

    grad_c = np.array([1, 1, 1])
    backward(c, grad=grad_c)

    assert_allclose(a.grad, [4, 5, 6])
    assert_allclose(b.grad, [1, 2, 3])


def test_sub_backward():
    a = Tensor([10, 20, 30], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=True)
    c = a - b

    grad_c = np.array([1, 2, 3])
    backward(c, grad=grad_c)

    assert_allclose(a.grad, [1, 2, 3])
    assert_allclose(b.grad, [-1, -2, -3])


def test_neg_backward():
    a = Tensor([1, 2, 3], requires_grad=True)
    c = -a

    grad_c = np.array([10, 20, 30])
    backward(c, grad=grad_c)

    assert_allclose(a.grad, [-10, -20, -30])


def test_pow_backward():
    a = Tensor([2, 3, 4], requires_grad=True)
    b = Tensor([3, 2, 1], requires_grad=True)
    c = a**b  # [8, 9, 4]

    grad_c = np.array([1, 1, 1])
    backward(c, grad=grad_c)

    assert_allclose(a.grad, [12, 6, 1])  # grad * b * a ** (b-1)
    expected_b_grad = [8 * np.log(2), 9 * np.log(3), 4 * np.log(4)]
    assert_allclose(b.grad, expected_b_grad)  # grad * (a ** b) * log(a)


def test_no_grad_propagation():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=False)  # No grad for b
    c = a * b

    backward(c, grad=np.ones_like(c.data))

    assert_allclose(a.grad, [4, 5, 6])
    assert b.grad is None  # b.grad was never initialized


def test_matmul_backward_2d():
    """
    Tests the backward pass for a standard 2D matrix multiplication.
    C = A @ B
    """

    a_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # shape (2, 3)
    b_data = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)  # shape (3, 2)

    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)

    # Upstream gradient (must match output shape)
    # Output C has shape (2, 2)
    grad_c = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    # Expected a.grad = grad_c @ b.T
    # b.T (shape 2, 3) = [[ 7,  9, 11],
    #                     [ 8, 10, 12]]
    # a.grad (shape 2, 3) = [[0.1, 0.2], [0.3, 0.4]] @ b.T
    expected_a_grad = np.array(
        [
            [
                0.1 * 7 + 0.2 * 8,
                0.1 * 9 + 0.2 * 10,
                0.1 * 11 + 0.2 * 12,
            ],  # [2.3, 2.9, 3.5]
            [
                0.3 * 7 + 0.4 * 8,
                0.3 * 9 + 0.4 * 10,
                0.3 * 11 + 0.4 * 12,
            ],  # [5.3, 6.7, 8.1]
        ]
    )

    # Expected b.grad = a.T @ grad_c
    # a.T (shape 3, 2) = [[1, 4],
    #                     [2, 5],
    #                     [3, 6]]
    # b.grad (shape 3, 2) = a.T @ [[0.1, 0.2], [0.3, 0.4]]
    expected_b_grad = np.array(
        [
            [1 * 0.1 + 4 * 0.3, 1 * 0.2 + 4 * 0.4],  # [1.3, 1.8]
            [2 * 0.1 + 5 * 0.3, 2 * 0.2 + 5 * 0.4],  # [1.7, 2.4]
            [3 * 0.1 + 6 * 0.3, 3 * 0.2 + 6 * 0.4],  # [2.1, 3.0]
        ]
    )

    c = a @ b  
    backward(c, grad=grad_c)

    assert a.grad.shape == a_data.shape
    assert_allclose(a.grad, expected_a_grad, rtol=1e-5)

    assert b.grad.shape == b_data.shape
    assert_allclose(b.grad, expected_b_grad, rtol=1e-5)


def test_matmul_with_no_grad():
    """Tests that gradients don't flow to tensors with requires_grad=False"""

    a_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_data = np.array([[5, 6], [7, 8]], dtype=np.float32)

    # 'a' requires grad, 'b' does not
    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=False)

    grad_c = np.array([[1, 1], [1, 1]], dtype=np.float32)

    # Expected a.grad = grad_c @ b.T
    expected_a_grad = np.array([[11, 15], [11, 15]])

    c = a @ b
    backward(c, grad=grad_c)

    assert a.grad is not None
    assert_allclose(a.grad, expected_a_grad)

    # Check b.grad (should be None since requires_grad=False)
    assert b.grad is None


# --- More complex tests ---


def test_complex_graph_chain_rule():
    # e = (a * b) + c
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    c = Tensor(4.0, requires_grad=True)

    d = a * b  # d = 6
    e = d + c  # e = 10

    backward(e)  # Start with grad=1.0

    # e.grad = 1
    # d.grad = de/dd * e.grad = 1 * 1 = 1
    # c.grad = de/dc * e.grad = 1 * 1 = 1
    assert_allclose(c.grad, 1.0)

    # a.grad = dd/da * d.grad = b * d.grad = 3.0 * 1 = 3.0
    # b.grad = dd/db * d.grad = a * d.grad = 2.0 * 1 = 2.0
    assert_allclose(a.grad, 3.0)
    assert_allclose(b.grad, 2.0)


def test_grad_accumulation():
    # c = (a * 2) + (a * 3)
    a = Tensor(5.0, requires_grad=True)
    b = a * 2.0  # b = 10
    c = a * 3.0  # c = 15
    d = b + c  # d = 25

    backward(d)

    # d.grad = 1
    # b.grad = dd/db * d.grad = 1 * 1 = 1
    # c.grad = dd/dc * d.grad = 1 * 1 = 1

    # a.grad should accumulate from both paths
    # Path 1 (via b): db/da * b.grad = 2.0 * 1 = 2.0
    # Path 2 (via c): dc/da * c.grad = 3.0 * 1 = 3.0
    # a.grad = 2.0 + 3.0 = 5.0
    assert_allclose(a.grad, 5.0)


def test_backward_with_broadcasting():
    # a (2, 2) + b (1, 2) -> c (2, 2)
    a = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
    b = Tensor(np.array([[10, 20]]), requires_grad=True)  # shape (1, 2)

    c = a + b
    # c.data = [[11, 22], [13, 24]]

    grad_c = np.ones((2, 2))
    backward(c, grad=grad_c)

    # a.grad should be same shape as grad_c
    assert_allclose(a.grad, [[1, 1], [1, 1]])

    # b.grad should be summed along axis 0
    # [1, 1] + [1, 1] -> [2, 2]
    # and have shape (1, 2)
    assert b.grad.shape == (1, 2)
    assert_allclose(b.grad, [[2, 2]])
