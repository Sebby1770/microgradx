"""
Mixed-precision utilities.

`autocast`: a context manager that flips a thread-local flag. Layers that
opt in (Linear, Conv2d, MatMul) cast their inputs to fp16 inside their
forward and cast back to fp32 on output.

`GradScaler`: scales the loss by `init_scale` before backward so fp16 grads
don't underflow, then unscales before optimizer.step. If any non-finite
gradient is detected we drop the step and halve the scale (standard recipe
matching torch.cuda.amp.GradScaler).

This is a *correct* but minimal implementation — we do all the math in fp32
under the hood (NumPy default), so the perf gain is theoretical here. The
plumbing is what matters: it ports straight to a CuPy / Triton backend.
"""
from __future__ import annotations
import threading
import contextlib
import numpy as np

from microgradx.tensor import Tensor


_state = threading.local()
_state.enabled = False
_state.dtype = np.float16


def is_autocast_enabled() -> bool:
    return getattr(_state, "enabled", False)


def autocast_dtype():
    return getattr(_state, "dtype", np.float16)


@contextlib.contextmanager
def autocast(enabled: bool = True, dtype=np.float16):
    prev_e, prev_d = is_autocast_enabled(), autocast_dtype()
    _state.enabled = enabled
    _state.dtype = dtype
    try:
        yield
    finally:
        _state.enabled = prev_e
        _state.dtype = prev_d


class GradScaler:
    """Scales loss → safe fp16 grads → unscale → step (skip if any NaN/Inf)."""

    def __init__(
        self,
        init_scale: float = 2.0 ** 10,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ):
        self.scale = float(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.enabled = enabled
        self._steps_since_growth = 0

    def scale_loss(self, loss: Tensor) -> Tensor:
        if not self.enabled:
            return loss
        return loss * self.scale

    def unscale_(self, optimizer):
        if not self.enabled:
            return
        inv = 1.0 / self.scale
        for p in optimizer.params:
            if p.grad is not None:
                p.grad *= inv

    def _has_nonfinite(self, optimizer) -> bool:
        for p in optimizer.params:
            if p.grad is not None and not np.all(np.isfinite(p.grad)):
                return True
        return False

    def step(self, optimizer):
        if not self.enabled:
            optimizer.step()
            return
        # Caller is expected to call `scaler.unscale_(opt)` before clipping
        # OR rely on us to do it here.
        if any(p.grad is not None and abs(p.grad).max() > 1.0
               for p in optimizer.params):
            # Only unscale if not already done — heuristic
            self.unscale_(optimizer)
        if self._has_nonfinite(optimizer):
            self._on_overflow()
            return
        optimizer.step()

    def update(self):
        """Adjust scale at end of step — grow periodically if no overflow."""
        if not self.enabled:
            return
        self._steps_since_growth += 1
        if self._steps_since_growth >= self.growth_interval:
            self.scale *= self.growth_factor
            self._steps_since_growth = 0

    def _on_overflow(self):
        self.scale = max(self.scale * self.backoff_factor, 1.0)
        self._steps_since_growth = 0
