"""Module freeze / unfreeze — frozen params get no grads / no updates."""
import numpy as np
import microgradx as mg
from microgradx import nn, optim


def test_freeze_sets_requires_grad_false():
    m = nn.Linear(4, 2)
    assert all(p.requires_grad for p in m.parameters())
    m.freeze()
    assert all(not p.requires_grad for p in m.parameters())
    m.unfreeze()
    assert all(p.requires_grad for p in m.parameters())


def test_requires_grad_underscore():
    m = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 1))
    m.requires_grad_(False)
    assert all(not p.requires_grad for p in m.parameters())
    m.requires_grad_(True)
    assert all(p.requires_grad for p in m.parameters())


def test_frozen_params_no_grad():
    m = nn.Linear(4, 2)
    m.freeze()
    x = mg.Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    y = m(x)
    y.sum().backward()
    for p in m.parameters():
        assert p.grad is None
    # Input still gets grad (it's not a parameter)
    assert x.grad is not None


def test_frozen_params_not_updated():
    m = nn.Linear(2, 1)
    # Snapshot weights
    w0 = m.weight.data.copy()
    b0 = m.bias.data.copy()
    m.freeze()
    opt = optim.SGD(list(m.parameters()), lr=0.1)
    x = mg.Tensor(np.ones((4, 2), dtype=np.float32))
    # Even if we force a grad, requires_grad=False means backward won't set it;
    # and optimizer skips None grads.
    target = mg.Tensor(np.zeros((4, 1), dtype=np.float32))
    pred = m(x)
    loss = ((pred - target) ** 2).mean()
    m.zero_grad()
    # loss may not require grad if all params frozen and x doesn't need grad
    # Use a path that still produces a scalar: add a free parameter path
    # Simpler: just verify freeze prevents grad then step is no-op.
    if loss.requires_grad:
        loss.backward()
    opt.step()
    np.testing.assert_array_equal(m.weight.data, w0)
    np.testing.assert_array_equal(m.bias.data, b0)


def test_partial_freeze_train_head():
    body = nn.Linear(4, 4)
    head = nn.Linear(4, 2)
    body.freeze()
    head.unfreeze()
    x = mg.Tensor(np.random.randn(2, 4).astype(np.float32))
    y = head(body(x))
    y.sum().backward()
    assert all(p.grad is None for p in body.parameters())
    assert any(p.grad is not None for p in head.parameters())
