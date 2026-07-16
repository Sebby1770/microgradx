import numpy as np

import microgradx as mg
from microgradx import nn


def test_checkpoint_matches_direct():
    """Checkpointing must be numerically transparent: identical forward output,
    identical input gradients, and identical parameter gradients."""
    np.random.seed(0)
    block = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))

    # --- direct ---
    x1 = mg.Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
    out1 = block(x1)
    (out1 * out1).sum().backward()
    g_x1 = x1.grad.copy()
    g_w1 = [p.grad.copy() for p in block.parameters()]

    # --- checkpointed (same weights, fresh grads) ---
    block.zero_grad()
    x2 = mg.Tensor(x1.numpy(), requires_grad=True)
    out2 = mg.checkpoint(lambda t: block(t), x2)
    (out2 * out2).sum().backward()

    assert np.allclose(out1.numpy(), out2.numpy())
    assert np.allclose(g_x1, x2.grad, atol=1e-5)
    for a, p in zip(g_w1, block.parameters()):
        assert np.allclose(a, p.grad, atol=1e-5)


def test_checkpoint_forward_is_grad_free():
    # The output of a checkpointed region still participates in autograd.
    x = mg.Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    y = mg.checkpoint(lambda t: t * 3.0, x)
    assert y.requires_grad
    y.sum().backward()
    assert np.allclose(x.grad, 3.0)


def test_checkpoint_rejects_multiple_outputs():
    x = mg.Tensor(np.zeros((2, 2), dtype=np.float32), requires_grad=True)
    try:
        mg.checkpoint(lambda t: (t, t), x)
    except TypeError:
        return
    raise AssertionError("expected TypeError for multi-output run_fn")
