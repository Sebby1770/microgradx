"""Early-stopping helper for training loops."""
from __future__ import annotations
from typing import Optional


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Typical usage::

        es = EarlyStopping(patience=5, mode="min")
        for epoch in range(max_epochs):
            val_loss = evaluate(...)
            if es.step(val_loss):
                break

    Parameters
    ----------
    patience : int
        Number of consecutive non-improving steps before stopping.
    mode : {"min", "max"}
        Whether a lower (``"min"``) or higher (``"max"``) metric is better.
    min_delta : float
        Minimum absolute change to count as an improvement.
    """

    def __init__(
        self,
        patience: int = 5,
        mode: str = "min",
        min_delta: float = 0.0,
    ):
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best: Optional[float] = None
        self.counter: int = 0
        self.stopped: bool = False
        self.best_epoch: int = 0
        self._epoch: int = 0

    def _is_improvement(self, metric: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return metric < self.best - self.min_delta
        return metric > self.best + self.min_delta

    def step(self, metric: float) -> bool:
        """Record ``metric`` and return ``True`` if training should stop."""
        metric = float(metric)
        self._epoch += 1
        if self._is_improvement(metric):
            self.best = metric
            self.best_epoch = self._epoch
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.stopped = True
            return True
        return False

    def reset(self) -> None:
        """Clear counters and best metric (reuse the same instance)."""
        self.best = None
        self.counter = 0
        self.stopped = False
        self.best_epoch = 0
        self._epoch = 0

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, mode={self.mode!r}, "
            f"best={self.best}, counter={self.counter})"
        )
