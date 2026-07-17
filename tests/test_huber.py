"""HuberLoss / SmoothL1Loss — finite loss and gradients."""
import numpy as np
import microgradx as mg
from microgradx import nn


def test_huber_loss_finite():
    pred = mg.Tensor(np.array([0.0, 1.0, 2.0, -3.0], dtype=np.float32), requires_grad=True)
    target = mg.Tensor(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    loss = nn.huber_loss(pred, target, delta=1.0)
    assert np.isfinite(float(loss.data))
    loss.backward()
    assert pred.grad is not None
    assert np.all(np.isfinite(pred.grad))


def test_huber_quadratic_region():
    # |e| = 0.5 < delta=1 → 0.5 * e²
    pred = mg.Tensor(np.array([0.5], dtype=np.float32), requires_grad=True)
    target = mg.Tensor(np.array([0.0], dtype=np.float32))
    loss = nn.huber_loss(pred, target, delta=1.0, reduction="sum")
    np.testing.assert_allclose(float(loss.data), 0.5 * 0.5 ** 2, atol=1e-6)
    loss.backward()
    # grad = e / delta = 0.5
    np.testing.assert_allclose(pred.grad, [0.5], atol=1e-6)


def test_huber_linear_region():
    # |e| = 3 > delta=1 → |e| - 0.5*delta = 2.5
    pred = mg.Tensor(np.array([3.0], dtype=np.float32), requires_grad=True)
    target = mg.Tensor(np.array([0.0], dtype=np.float32))
    loss = nn.huber_loss(pred, target, delta=1.0, reduction="sum")
    np.testing.assert_allclose(float(loss.data), 2.5, atol=1e-6)
    loss.backward()
    np.testing.assert_allclose(pred.grad, [1.0], atol=1e-6)


def test_smooth_l1_module():
    crit = nn.SmoothL1Loss(beta=1.0, reduction="mean")
    pred = mg.Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
    target = mg.Tensor(np.random.randn(4, 3).astype(np.float32))
    loss = crit(pred, target)
    loss.backward()
    assert np.isfinite(float(loss.data))
    assert np.all(np.isfinite(pred.grad))


def test_huber_module_reduction_none():
    crit = nn.HuberLoss(delta=1.0, reduction="none")
    pred = mg.Tensor(np.array([0.0, 2.0], dtype=np.float32), requires_grad=True)
    target = mg.Tensor(np.array([0.0, 0.0], dtype=np.float32))
    loss = crit(pred, target)
    assert loss.shape == (2,)
