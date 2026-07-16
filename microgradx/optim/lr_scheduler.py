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


class OneCycleLR(_LRScheduler):
    """1cycle learning-rate policy (Smith 2018).

    Warms up from ``max_lr / div_factor`` to ``max_lr`` over the first
    ``pct_start * total_steps`` steps, then anneals down to
    ``max_lr / (div_factor * final_div_factor)`` over the remainder.

    Annealing is cosine (``anneal_strategy="cos"``) or linear
    (``"linear"``). Step this scheduler **once per optimizer step**
    (typically once per batch).

        sched = OneCycleLR(opt, max_lr=1e-2, total_steps=1000)
        for batch in loader:
            ...
            opt.step()
            sched.step()
    """

    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0.0 < pct_start < 1.0:
            raise ValueError("pct_start must be in (0, 1)")
        if anneal_strategy not in ("cos", "linear"):
            raise ValueError("anneal_strategy must be 'cos' or 'linear'")
        if div_factor <= 0 or final_div_factor <= 0:
            raise ValueError("div_factor and final_div_factor must be positive")

        self.max_lr = float(max_lr)
        self.total_steps = int(total_steps)
        self.pct_start = float(pct_start)
        self.anneal_strategy = anneal_strategy
        self.div_factor = float(div_factor)
        self.final_div_factor = float(final_div_factor)

        self.initial_lr = self.max_lr / self.div_factor
        self.min_lr = self.initial_lr / self.final_div_factor
        # Phase end steps (inclusive), matching PyTorch: last training step
        # lands exactly on min_lr at last_epoch == total_steps - 1.
        self.step_size_up = max(0, int(self.pct_start * self.total_steps) - 1)
        self.step_size_down = (self.total_steps - 1) - self.step_size_up

        # Capture base_lr as max_lr so state_dict stays consistent; the
        # optimizer's actual rate is rewritten on every step.
        if "lr" not in optimizer.defaults:
            raise ValueError("optimizer has no 'lr' in defaults")
        self.optimizer = optimizer
        self.base_lr = self.max_lr
        self.last_epoch = last_epoch
        self.step()

    @staticmethod
    def _annealing_cos(start: float, end: float, pct: float) -> float:
        cos_out = math.cos(math.pi * pct) + 1.0
        return end + (start - end) / 2.0 * cos_out

    @staticmethod
    def _annealing_linear(start: float, end: float, pct: float) -> float:
        return start + (end - start) * pct

    def _anneal(self, start: float, end: float, pct: float) -> float:
        pct = min(max(pct, 0.0), 1.0)
        if self.anneal_strategy == "cos":
            return self._annealing_cos(start, end, pct)
        return self._annealing_linear(start, end, pct)

    def get_lr(self) -> float:
        step = self.last_epoch
        if step >= self.total_steps:
            return self.min_lr
        if step <= self.step_size_up:
            denom = self.step_size_up if self.step_size_up > 0 else 1
            pct = step / denom
            return self._anneal(self.initial_lr, self.max_lr, pct)
        # Anneal phase: step_size_up < step <= total_steps - 1
        down = step - self.step_size_up
        denom = self.step_size_down if self.step_size_down > 0 else 1
        pct = down / denom
        return self._anneal(self.max_lr, self.min_lr, pct)

    def state_dict(self) -> dict:
        return {
            "last_epoch": self.last_epoch,
            "base_lr": self.base_lr,
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
            "pct_start": self.pct_start,
            "anneal_strategy": self.anneal_strategy,
            "div_factor": self.div_factor,
            "final_div_factor": self.final_div_factor,
        }

    def load_state_dict(self, sd: dict):
        self.last_epoch = sd["last_epoch"]
        self.base_lr = sd["base_lr"]
        self.max_lr = sd.get("max_lr", self.max_lr)
        self.total_steps = sd.get("total_steps", self.total_steps)
        self.pct_start = sd.get("pct_start", self.pct_start)
        self.anneal_strategy = sd.get("anneal_strategy", self.anneal_strategy)
        self.div_factor = sd.get("div_factor", self.div_factor)
        self.final_div_factor = sd.get("final_div_factor", self.final_div_factor)
        self.initial_lr = self.max_lr / self.div_factor
        self.min_lr = self.initial_lr / self.final_div_factor
        self.step_size_up = max(0, int(self.pct_start * self.total_steps) - 1)
        self.step_size_down = (self.total_steps - 1) - self.step_size_up

