"""CosineAnnealingWarmRestarts — LR decays then resets each cycle."""
import math

import numpy as np

import microgradx as mg
from microgradx import optim


def fresh_opt(lr=0.1):
    p = mg.Tensor([1.0], requires_grad=True)
    return optim.SGD([p], lr=lr)


def test_warm_restarts_resets():
    opt = fresh_opt(lr=0.1)
    T_0 = 4
    s = optim.CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=1, eta_min=0.0)
    lrs = [opt.defaults["lr"]]  # epoch 0 → base_lr
    for _ in range(T_0 * 2):
        s.step()
        lrs.append(opt.defaults["lr"])

    # At start of each cycle LR ≈ base_lr
    assert np.isclose(lrs[0], 0.1)
    assert np.isclose(lrs[T_0], 0.1, atol=1e-6)  # restart
    # Mid-cycle should be lower than base
    assert lrs[T_0 // 2] < 0.1
    # End of first cycle (T_cur = T_0-1) near eta_min
    expected_end = 0.1 * (1 + math.cos(math.pi * (T_0 - 1) / T_0)) / 2
    assert np.isclose(lrs[T_0 - 1], expected_end, atol=1e-6)


def test_warm_restarts_t_mult():
    opt = fresh_opt(lr=1.0)
    s = optim.CosineAnnealingWarmRestarts(opt, T_0=2, T_mult=2, eta_min=0.0)
    # Cycle lengths: 2, then 4, then 8, ...
    # After init: T_i=2, T_cur=0
    assert s.T_i == 2
    s.step()  # T_cur=1
    s.step()  # T_cur=0, T_i=4 (restart)
    assert s.T_i == 4
    assert s.T_cur == 0
    assert np.isclose(opt.defaults["lr"], 1.0)


def test_warm_restarts_eta_min():
    opt = fresh_opt(lr=1.0)
    s = optim.CosineAnnealingWarmRestarts(opt, T_0=4, eta_min=0.1)
    # T_cur=0 → lr = base_lr
    assert np.isclose(opt.defaults["lr"], 1.0)
    for _ in range(3):
        s.step()
    # T_cur=3, T_i=4 → near the bottom of the half-cosine
    expected = 0.1 + (1.0 - 0.1) * (1 + math.cos(math.pi * 3 / 4)) / 2
    assert np.isclose(opt.defaults["lr"], expected, atol=1e-6)
    assert opt.defaults["lr"] < 0.3  # clearly annealed toward eta_min
