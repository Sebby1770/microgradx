"""BCEWithLogitsLoss and BCELoss — shapes and finite grads."""
import numpy as np
import microgradx as mg
from microgradx import nn


def test_bce_with_logits_shape_and_value():
    logits = mg.Tensor(np.array([[0.0, 2.0], [-1.0, 0.5]], dtype=np.float32),
                       requires_grad=True)
    target = mg.Tensor(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    loss = nn.binary_cross_entropy_with_logits(logits, target)
    assert loss.shape == ()
    # Compare to manual stable formula
    x = logits.numpy()
    y = target.numpy()
    expected = (np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))).mean()
    np.testing.assert_allclose(float(loss.data), expected, atol=1e-5)


def test_bce_with_logits_finite_grads():
    logits = mg.Tensor(np.random.randn(4, 3).astype(np.float32) * 3, requires_grad=True)
    target = mg.Tensor(np.random.rand(4, 3).astype(np.float32))
    loss = nn.BCEWithLogitsLoss()(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert np.all(np.isfinite(logits.grad))
    assert logits.grad.shape == logits.shape


def test_bce_with_logits_reduction_none():
    logits = mg.Tensor(np.zeros((2, 2), dtype=np.float32), requires_grad=True)
    target = mg.Tensor(np.ones((2, 2), dtype=np.float32))
    loss = nn.binary_cross_entropy_with_logits(logits, target, reduction="none")
    assert loss.shape == (2, 2)
    # x=0, y=1 → max(0,0) - 0 + log(1+1) = log(2)
    np.testing.assert_allclose(loss.numpy(), np.log(2), atol=1e-5)


def test_bce_loss_probs():
    pred = mg.Tensor(np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
                     requires_grad=True)
    target = mg.Tensor(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    loss = nn.BCELoss()(pred, target)
    assert loss.shape == ()
    loss.backward()
    assert pred.grad is not None
    assert np.all(np.isfinite(pred.grad))


def test_bce_large_logits_stable():
    """Large positive/negative logits must not produce NaN/Inf."""
    logits = mg.Tensor(np.array([[50.0, -50.0]], dtype=np.float32), requires_grad=True)
    target = mg.Tensor(np.array([[1.0, 0.0]], dtype=np.float32))
    loss = nn.binary_cross_entropy_with_logits(logits, target)
    assert np.isfinite(float(loss.data))
    loss.backward()
    assert np.all(np.isfinite(logits.grad))
