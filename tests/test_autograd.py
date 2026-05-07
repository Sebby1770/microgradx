"""Comprehensive gradient checking across all primitive ops."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import microgradx as mg
from microgradx import gradcheck

np.random.seed(0)


def test_arith_ops():
    x = mg.Tensor(np.random.randn(3, 4), requires_grad=True)
    y = mg.Tensor(np.random.randn(3, 4), requires_grad=True)
    assert gradcheck(lambda a, b: (a + b - a*b/(b+2)).sum(), [x, y])


def test_pow():
    x = mg.Tensor(np.random.rand(3, 4) + 0.5, requires_grad=True)
    assert gradcheck(lambda x: (x ** 1.7).sum(), [x])


def test_log_exp_sqrt():
    x = mg.Tensor(np.random.rand(3, 4) + 0.5, requires_grad=True)
    assert gradcheck(lambda x: (x.log() + x.exp() + x.sqrt()).sum(), [x])


def test_relu_tanh_sigmoid():
    x = mg.Tensor(np.random.randn(3, 4), requires_grad=True)
    assert gradcheck(lambda x: (x.relu() + x.tanh() + x.sigmoid()).sum(), [x])


def test_softmax():
    x = mg.Tensor(np.random.randn(2, 5), requires_grad=True)
    assert gradcheck(lambda x: (x.softmax(-1) * 7.0).sum(), [x])
    assert gradcheck(lambda x: x.log_softmax(-1).sum(), [x])


def test_matmul_2d_and_batched():
    a = mg.Tensor(np.random.randn(3, 4), requires_grad=True)
    b = mg.Tensor(np.random.randn(4, 5), requires_grad=True)
    assert gradcheck(lambda a, b: (a @ b).sum(), [a, b])

    a = mg.Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    b = mg.Tensor(np.random.randn(2, 4, 5), requires_grad=True)
    assert gradcheck(lambda a, b: (a @ b).sum(), [a, b])


def test_reductions_with_axis():
    x = mg.Tensor(np.random.randn(2, 3, 4, 5), requires_grad=True)
    for axis in [None, 0, 1, 2, 3, -1]:
        for keepdims in [False, True]:
            ok = gradcheck(lambda x, ax=axis, k=keepdims:
                           x.sum(axis=ax, keepdims=k).sum(), [x])
            assert ok, f"sum axis={axis} keepdims={keepdims} failed"
            ok = gradcheck(lambda x, ax=axis, k=keepdims:
                           x.mean(axis=ax, keepdims=k).sum(), [x])
            assert ok, f"mean axis={axis} keepdims={keepdims} failed"


def test_max_axis():
    # Use widely-spaced values so ties don't muddy gradient
    x = mg.Tensor(np.random.randn(3, 4, 5) * 10, requires_grad=True)
    assert gradcheck(lambda x: x.max(axis=1).sum(), [x])


def test_reshape_transpose():
    x = mg.Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    assert gradcheck(lambda x: x.reshape(6, 4).transpose().sum(), [x])
    assert gradcheck(lambda x: x.permute(2, 0, 1).sum(), [x])


def test_getitem():
    x = mg.Tensor(np.random.randn(5, 4), requires_grad=True)
    assert gradcheck(lambda x: x[1:4, ::2].sum(), [x])


def test_chained_complex():
    a = mg.Tensor(np.random.randn(3, 4) * 0.5, requires_grad=True)
    b = mg.Tensor(np.random.randn(4, 2) * 0.5, requires_grad=True)
    fn = lambda a, b: ((a @ b).relu() + 1).log().sum(axis=1).mean()
    assert gradcheck(fn, [a, b])
