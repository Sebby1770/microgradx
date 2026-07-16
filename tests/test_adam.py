"""Classic Adam — convergence on a simple linear / quadratic problem."""
import numpy as np
import microgradx as mg
from microgradx import nn, optim


def test_adam_steps_reduce_loss_on_linear():
    np.random.seed(0)
    model = nn.Linear(4, 1, bias=True)
    # Target: y = sum(x) roughly
    X = np.random.randn(32, 4).astype(np.float32)
    y = X.sum(axis=1, keepdims=True).astype(np.float32)

    opt = optim.Adam(model.parameters(), lr=0.05)

    def loss_val():
        pred = model(mg.Tensor(X))
        return float(((pred - mg.Tensor(y)) ** 2).mean().data)

    losses = []
    for _ in range(80):
        pred = model(mg.Tensor(X))
        loss = ((pred - mg.Tensor(y)) ** 2).mean()
        model.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    assert losses[-1] < losses[0] * 0.5, (
        f"Adam did not reduce loss enough: {losses[0]:.4f} → {losses[-1]:.4f}"
    )
    assert losses[-1] < 0.5


def test_adam_quadratic_converges():
    x = mg.Tensor(np.array([5.0, -3.0, 7.0], dtype=np.float32), requires_grad=True)
    target = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    opt = optim.Adam([x], lr=0.1)
    for _ in range(500):
        x.zero_grad()
        loss = ((x - mg.Tensor(target)) ** 2).sum()
        loss.backward()
        opt.step()
    err = float(np.abs(x.numpy() - target).max())
    assert err < 1e-2, f"Adam quadratic err={err}"


def test_adam_weight_decay_l2_style():
    # With large weight_decay and zero grad, parameters should shrink toward 0.
    p = mg.Tensor(np.array([1.0, -1.0], dtype=np.float32), requires_grad=True)
    opt = optim.Adam([p], lr=0.1, weight_decay=1.0)
    p.grad = np.zeros_like(p.data)
    opt.step()
    # g = 0 + wd * p = p; first step m≈0.1p, v≈0.001 p² → update > 0 so |p| shrinks
    assert float(np.abs(p.numpy()).max()) < 1.0
