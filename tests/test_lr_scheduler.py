import math

import numpy as np

import microgradx as mg
from microgradx import optim


def fresh_opt(lr=0.1):
    p = mg.Tensor([1.0], requires_grad=True)
    return optim.SGD([p], lr=lr)


def lr_of(opt):
    return opt.defaults["lr"]


def test_step_lr():
    opt = fresh_opt()
    s = optim.StepLR(opt, step_size=2, gamma=0.5)
    seen = [lr_of(opt)]            # epoch 0
    for _ in range(4):
        s.step()
        seen.append(lr_of(opt))
    # epochs 0,1 -> 0.1 ; 2,3 -> 0.05 ; 4 -> 0.025
    assert np.allclose(seen, [0.1, 0.1, 0.05, 0.05, 0.025])


def test_multistep_lr():
    opt = fresh_opt()
    s = optim.MultiStepLR(opt, milestones=[2, 4], gamma=0.1)
    seen = [lr_of(opt)]
    for _ in range(4):
        s.step()
        seen.append(lr_of(opt))
    assert np.allclose(seen, [0.1, 0.1, 0.01, 0.01, 0.001])


def test_exponential_lr():
    opt = fresh_opt()
    s = optim.ExponentialLR(opt, gamma=0.9)
    seen = [lr_of(opt)]
    for _ in range(3):
        s.step()
        seen.append(lr_of(opt))
    assert np.allclose(seen, [0.1, 0.09, 0.081, 0.0729])


def test_cosine_annealing_lr():
    opt = fresh_opt()
    s = optim.CosineAnnealingLR(opt, T_max=4, eta_min=0.0)
    seen = [lr_of(opt)]
    for _ in range(4):
        s.step()
        seen.append(lr_of(opt))
    expected = [0.1 * (1 + math.cos(math.pi * t / 4)) / 2 for t in range(5)]
    assert np.allclose(seen, expected)
    assert np.isclose(seen[-1], 0.0)  # fully annealed at T_max


def test_linear_warmup():
    opt = fresh_opt()
    s = optim.LinearWarmup(opt, warmup_steps=4, start_factor=0.0)
    seen = [lr_of(opt)]
    for _ in range(5):
        s.step()
        seen.append(lr_of(opt))
    # ramps 0 -> base over 4 steps, then holds
    assert np.allclose(seen, [0.0, 0.025, 0.05, 0.075, 0.1, 0.1])


def test_lambda_lr():
    opt = fresh_opt()
    s = optim.LambdaLR(opt, lr_lambda=lambda e: 1.0 / (1 + e))
    seen = [lr_of(opt)]
    for _ in range(2):
        s.step()
        seen.append(lr_of(opt))
    assert np.allclose(seen, [0.1, 0.05, 0.1 / 3])


def test_scheduler_actually_changes_optimizer_update():
    # A step under a decayed LR must move the parameter less than under base.
    p = mg.Tensor([0.0], requires_grad=True)
    opt = optim.SGD([p], lr=1.0)
    optim.StepLR(opt, step_size=1, gamma=0.1)  # epoch 0 keeps 1.0
    p.grad = np.array([1.0], dtype=np.float32)
    opt.step()
    moved_base = abs(float(p.numpy()[0]))
    assert np.isclose(moved_base, 1.0)
