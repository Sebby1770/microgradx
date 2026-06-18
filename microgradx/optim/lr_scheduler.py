"""Learning-rate schedulers.

A scheduler wraps an optimizer and rewrites ``optimizer.defaults["lr"]`` on
every ``step()``. Both built-in optimizers read ``lr`` fresh from
``defaults`` inside their own ``step()``, so a scheduler change takes effect
on the very next update.

Usage (PyTorch order: optimizer step, then scheduler step, once per epoch or
per iteration):

    sched = CosineAnnealingLR(opt, T_max=100)
    for epoch in range(100):
        train_one_epoch()
        sched.step()
        print(sched.get_last_lr())

Every scheduler computes its rate from the captured ``base_lr`` rather than
by mutating the current value, so they compose predictably and never drift.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence


class _LRScheduler:
    """Base scheduler. Subclasses implement :meth:`get_lr`."""

    def __init__(self, optimizer, last_epoch: int = -1):
        if "lr" not in optimizer.defaults:
            raise ValueError("optimizer has no 'lr' in defaults")
        self.optimizer = optimizer
        self.base_lr = float(optimizer.defaults["lr"])
        self.last_epoch = last_epoch
        # Mirror PyTorch: stepping once in __init__ sets the epoch-0 rate.
        self.step()

    def get_lr(self) -> float:
        raise NotImplementedError

    def get_last_lr(self) -> float:
        return float(self.optimizer.defaults["lr"])

    def step(self):
        self.last_epoch += 1
        lr = self.get_lr()
        self.optimizer.defaults["lr"] = lr

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch, "base_lr": self.base_lr}

    def load_state_dict(self, sd: dict):
        self.last_epoch = sd["last_epoch"]
        self.base_lr = sd["base_lr"]


class LambdaLR(_LRScheduler):
    """``lr = base_lr * lr_lambda(epoch)``."""

    def __init__(self, optimizer, lr_lambda: Callable[[int], float],
                 last_epoch: int = -1):
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * float(self.lr_lambda(self.last_epoch))


class StepLR(_LRScheduler):
    """Decay by ``gamma`` every ``step_size`` epochs."""

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1,
                 last_epoch: int = -1):
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class MultiStepLR(_LRScheduler):
    """Decay by ``gamma`` once the epoch passes each milestone."""

    def __init__(self, optimizer, milestones: Sequence[int], gamma: float = 0.1,
                 last_epoch: int = -1):
        self.milestones = sorted(int(m) for m in milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        n = sum(1 for m in self.milestones if m <= self.last_epoch)
        return self.base_lr * self.gamma ** n


class ExponentialLR(_LRScheduler):
    """Multiply the rate by ``gamma`` every epoch."""

    def __init__(self, optimizer, gamma: float, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        return self.base_lr * self.gamma ** self.last_epoch


class CosineAnnealingLR(_LRScheduler):
    """Cosine decay from ``base_lr`` to ``eta_min`` over ``T_max`` epochs."""

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0,
                 last_epoch: int = -1):
        if T_max <= 0:
            raise ValueError("T_max must be positive")
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        t = min(self.last_epoch, self.T_max)
        cos = (1 + math.cos(math.pi * t / self.T_max)) / 2
        return self.eta_min + (self.base_lr - self.eta_min) * cos


class LinearWarmup(_LRScheduler):
    """Ramp linearly from ``start_factor*base_lr`` to ``base_lr`` over
    ``warmup_steps`` steps, then hold at ``base_lr``.

    Often paired by hand with a decay schedule: warm up for the first N
    steps, then switch to cosine.
    """

    def __init__(self, optimizer, warmup_steps: int, start_factor: float = 0.0,
                 last_epoch: int = -1):
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        self.warmup_steps = warmup_steps
        self.start_factor = start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch >= self.warmup_steps:
            return self.base_lr
        frac = self.last_epoch / self.warmup_steps
        factor = self.start_factor + (1.0 - self.start_factor) * frac
        return self.base_lr * factor
