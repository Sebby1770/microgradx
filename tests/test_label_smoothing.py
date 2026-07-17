"""Label-smoothing CrossEntropy."""
import numpy as np
import microgradx as mg
from microgradx import nn


def test_cross_entropy_with_label_smoothing_runs():
    logits = mg.Tensor(np.random.randn(5, 4).astype(np.float32), requires_grad=True)
    target = np.array([0, 1, 2, 3, 1], dtype=np.int64)
    loss = nn.cross_entropy(logits, target, label_smoothing=0.1)
    assert loss.shape == ()
    assert np.isfinite(float(loss.data))
    loss.backward()
    assert logits.grad is not None
    assert np.all(np.isfinite(logits.grad))
    assert logits.grad.shape == (5, 4)


def test_cross_entropy_module_label_smoothing():
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    logits = mg.Tensor(np.random.randn(3, 6).astype(np.float32), requires_grad=True)
    target = np.array([0, 2, 5], dtype=np.int64)
    loss = crit(logits, target)
    loss.backward()
    assert np.all(np.isfinite(logits.grad))


def test_label_smoothing_zero_matches_hard():
    logits_data = np.random.randn(4, 5).astype(np.float32)
    target = np.array([0, 1, 2, 3], dtype=np.int64)
    a = mg.Tensor(logits_data.copy(), requires_grad=True)
    b = mg.Tensor(logits_data.copy(), requires_grad=True)
    la = nn.cross_entropy(a, target, label_smoothing=0.0)
    lb = nn.cross_entropy(b, target)  # default 0
    np.testing.assert_allclose(float(la.data), float(lb.data), atol=1e-6)


def test_label_smoothing_increases_loss_on_confident():
    # One-hot confident prediction on true class: hard CE ~ 0, smoothed > 0
    logits = mg.Tensor(
        np.array([[10.0, 0.0, 0.0]], dtype=np.float32), requires_grad=True
    )
    target = np.array([0], dtype=np.int64)
    hard = float(nn.cross_entropy(logits, target, label_smoothing=0.0).data)
    smooth = float(
        nn.cross_entropy(
            mg.Tensor(np.array([[10.0, 0.0, 0.0]], dtype=np.float32)),
            target,
            label_smoothing=0.2,
        ).data
    )
    assert smooth > hard
