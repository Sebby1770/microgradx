"""RAdam — simple linear fit decreases loss."""
import numpy as np
import microgradx as mg
from microgradx import nn, optim


def test_radam_linear_fit_decreases_loss():
    rng = np.random.default_rng(0)
    # y ≈ 2x + 1
    x_np = rng.normal(size=(64, 1)).astype(np.float32)
    y_np = (2.0 * x_np + 1.0 + 0.05 * rng.normal(size=x_np.shape)).astype(np.float32)

    model = nn.Linear(1, 1)
    opt = optim.RAdam(model.parameters(), lr=5e-2)
    crit = nn.MSELoss()

    def batch_loss():
        pred = model(mg.Tensor(x_np))
        return crit(pred, mg.Tensor(y_np))

    losses = []
    for _ in range(80):
        loss = batch_loss()
        model.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))

    assert losses[-1] < losses[0] * 0.5
    assert np.isfinite(losses[-1])


def test_radam_step_no_nan():
    p = mg.Tensor(np.array([1.0, -1.0], dtype=np.float32), requires_grad=True)
    opt = optim.RAdam([p], lr=1e-2)
    for _ in range(5):
        loss = (p * p).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert np.all(np.isfinite(p.data))
