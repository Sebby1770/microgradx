"""Optimizers — convergence on a tiny convex problem + dtype safety."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import microgradx as mg
from microgradx import nn, optim

np.random.seed(0)


def _quadratic(opt_cls, **kwargs):
    """Minimise f(x) = (x - target)² with given optimizer.  Returns final |x-target|."""
    x = mg.Tensor(np.array([5.0, -3.0, 7.0], dtype=np.float32), requires_grad=True)
    target = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    opt = opt_cls([x], **kwargs)
    for _ in range(2000):
        x.zero_grad()
        loss = ((x - mg.Tensor(target)) ** 2).sum()
        loss.backward()
        opt.step()
    return float(np.abs(x.numpy() - target).max())


def test_sgd_converges():
    err = _quadratic(optim.SGD, lr=0.05)
    assert err < 1e-3, f"SGD didn't converge: err={err}"


def test_sgd_momentum_converges():
    err = _quadratic(optim.SGD, lr=0.01, momentum=0.9)
    assert err < 1e-3


def test_adamw_converges():
    err = _quadratic(optim.AdamW, lr=0.05)
    assert err < 1e-2


def test_lion_converges():
    err = _quadratic(optim.Lion, lr=0.01)
    assert err < 1e-1


def test_dtype_preserved_after_steps():
    model = nn.Linear(4, 4)
    x = mg.Tensor(np.random.randn(8, 4))
    y = np.zeros(8, dtype=np.int64)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(20):
        loss = nn.cross_entropy(model(x), y)
        model.zero_grad()
        loss.backward()
        opt.step()
    assert model.weight.dtype == np.float32
    assert model.bias.dtype == np.float32


def test_grad_clip_norm():
    p = mg.Tensor(np.array([3.0, 4.0]), requires_grad=True)
    p.grad = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5
    norm = optim.clip_grad_norm_([p], max_norm=1.0)
    np.testing.assert_allclose(norm, 5.0, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(p.grad), 1.0, atol=1e-4)


def test_grad_clip_value():
    p = mg.Tensor(np.array([1.0, 2.0]), requires_grad=True)
    p.grad = np.array([10.0, -5.0], dtype=np.float32)
    optim.clip_grad_value_([p], 1.0)
    np.testing.assert_array_equal(p.grad, [1.0, -1.0])
