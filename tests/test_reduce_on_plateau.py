"""ReduceLROnPlateau — reduce LR when metric plateaus."""
import numpy as np
import microgradx as mg
from microgradx import optim


def fresh_opt(lr=1.0):
    p = mg.Tensor([1.0], requires_grad=True)
    return optim.SGD([p], lr=lr)


def test_reduce_on_plateau_min_mode():
    opt = fresh_opt(lr=1.0)
    sched = optim.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=2, threshold=0.0
    )
    # First call sets best
    sched.step(1.0)
    assert np.isclose(opt.defaults["lr"], 1.0)
    # Improving — no reduce
    sched.step(0.9)
    assert np.isclose(opt.defaults["lr"], 1.0)
    # Plateau: 2 non-improving, then reduce on 3rd bad (patience=2 → reduce when >2)
    sched.step(0.95)  # bad 1
    sched.step(0.96)  # bad 2
    assert np.isclose(opt.defaults["lr"], 1.0)
    sched.step(0.97)  # bad 3 > patience → reduce
    assert np.isclose(opt.defaults["lr"], 0.5)


def test_reduce_on_plateau_max_mode():
    opt = fresh_opt(lr=0.1)
    sched = optim.ReduceLROnPlateau(
        opt, mode="max", factor=0.1, patience=1, threshold=0.0
    )
    sched.step(0.5)   # best
    sched.step(0.4)   # bad 1
    assert np.isclose(opt.defaults["lr"], 0.1)
    sched.step(0.3)   # bad 2 > patience → reduce
    assert np.isclose(opt.defaults["lr"], 0.01)


def test_reduce_on_plateau_min_lr():
    opt = fresh_opt(lr=0.1)
    sched = optim.ReduceLROnPlateau(
        opt, mode="min", factor=0.1, patience=0, threshold=0.0, min_lr=0.05
    )
    sched.step(1.0)
    sched.step(1.1)  # bad → reduce but floored at min_lr
    assert np.isclose(opt.defaults["lr"], 0.05)
    # Further reductions stay at floor
    sched.step(1.2)
    assert np.isclose(opt.defaults["lr"], 0.05)


def test_reduce_on_plateau_cooldown():
    opt = fresh_opt(lr=1.0)
    sched = optim.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=0, threshold=0.0, cooldown=2
    )
    sched.step(1.0)
    sched.step(1.1)  # reduce to 0.5, start cooldown
    assert np.isclose(opt.defaults["lr"], 0.5)
    # During cooldown, non-improving metrics should not reduce again
    sched.step(1.2)
    sched.step(1.3)
    assert np.isclose(opt.defaults["lr"], 0.5)
    # Cooldown finished; next bad epoch reduces
    sched.step(1.4)
    assert np.isclose(opt.defaults["lr"], 0.25)


def test_get_last_lr():
    opt = fresh_opt(lr=0.2)
    sched = optim.ReduceLROnPlateau(opt, patience=0, threshold=0.0)
    sched.step(1.0)
    assert np.isclose(sched.get_last_lr(), 0.2)
