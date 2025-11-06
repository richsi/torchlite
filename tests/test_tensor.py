# tests/test_tesnor.py
import numpy as np
from torchlite.autograd.tensor import Tensor


def test_tensor_creation():
    t = Tensor([1, 2, 3])
    assert np.array_equal(t.data, np.array([1, 2, 3], dtype=float))


def test_tensor_shape():
    t = Tensor([[1, 2], [3, 4]])
    assert t.shape == (2, 2)


def test_tensor_add():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    result = t1 + t2
    assert np.array_equal(result.data, np.array([5, 7, 9]))


def test_tensor_mul():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    result = t1 * t2
    assert np.array_equal(result.data, np.array([4, 10, 18]))


def test_tensor_pow():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([2, 2, 2])
    result = t1**t2
    assert np.array_equal(result.data, np.array([1, 4, 9]))


def test_tensor_sub():
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    result = t2 - t1
    assert np.array_equal(result.data, np.array([3, 3, 3]))


def test_tensor_neg():
    t1 = Tensor([1, 2, 3])
    t_neg = -t1
    assert np.array_equal(t_neg.data, np.array([-1, -2, -3]))


def test_reverse_op():
    t = Tensor([1, 2, 3])
    out = 1 + t
    assert np.array_equal(out.data, np.array([2, 3, 4]))
    out = 2 * t
    assert np.array_equal(out.data, np.array([2, 4, 6]))
    out = 3 - t
    assert np.array_equal(out.data, np.array([2, 1, 0]))
