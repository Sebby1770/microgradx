"""OneCycleLR — warmup then anneal."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import microgradx as mg
from microgradx import optim


def fresh_opt(lr=0.1):
    p = mg.Tensor([1.0], requires_grad=True)
    return optim.SGD([p], lr=lr)


def test_onecycle_goes_up_then_down():
    max_lr = 1.0
    total = 100
    opt = fresh_opt(lr=0.01)
    sched = optim.OneCycleLR(
        opt, max_lr=max_lr, total_steps=total, pct_start=0.3,
        anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4,
    )
    lrs = [sched.get_last_lr()]
    for _ in range(total - 1):
        sched.step()
        lrs.append(sched.get_last_lr())

    peak_idx = int(np.argmax(lrs))
    peak = lrs[peak_idx]
    # Peak near max_lr and somewhere in the first half-ish (warmup).
    assert abs(peak - max_lr) < 1e-6
    assert peak_idx < total * 0.5
    # First LR is the initial (low) rate.
    assert lrs[0] < max_lr * 0.5
    # Early steps rise toward peak.
    assert lrs[max(1, peak_idx // 2)] < peak
    # Late steps fall below peak and below the initial rate.
    assert lrs[-1] < peak
    assert lrs[-1] < lrs[0]


def test_onecycle_linear_anneal():
    opt = fresh_opt()
    sched = optim.OneCycleLR(
        opt, max_lr=0.5, total_steps=20, pct_start=0.25,
        anneal_strategy="linear", div_factor=10.0, final_div_factor=100.0,
    )
    lrs = [sched.get_last_lr()]
    for _ in range(19):
        sched.step()
        lrs.append(sched.get_last_lr())
    peak_idx = int(np.argmax(lrs))
    assert abs(lrs[peak_idx] - 0.5) < 1e-6
    # Monotone non-decreasing up to peak, non-increasing after.
    for i in range(1, peak_idx + 1):
        assert lrs[i] >= lrs[i - 1] - 1e-12
    for i in range(peak_idx + 1, len(lrs)):
        assert lrs[i] <= lrs[i - 1] + 1e-12


def test_onecycle_initial_and_final_rates():
    max_lr = 1.0
    div = 25.0
    final_div = 1e4
    opt = fresh_opt()
    sched = optim.OneCycleLR(
        opt, max_lr=max_lr, total_steps=50, pct_start=0.3,
        div_factor=div, final_div_factor=final_div,
    )
    initial = max_lr / div
    final = initial / final_div
    assert np.isclose(sched.get_last_lr(), initial)
    for _ in range(49):
        sched.step()
    assert np.isclose(sched.get_last_lr(), final, rtol=1e-3, atol=1e-8)
