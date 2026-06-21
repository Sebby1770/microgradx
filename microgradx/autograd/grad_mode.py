"""Global autograd on/off switch + `no_grad` / `enable_grad` context managers.

By default every op records the graph needed for `backward()`. During
inference that bookkeeping is wasted work — and worse, it keeps the whole
forward graph alive in memory. Wrapping a forward pass in `no_grad()` makes
`Function.apply` skip graph construction entirely:

    model.eval()
    with mg.no_grad():
        logits = model(x)          # no graph built, no memory retained

The objects double as decorators:

    @mg.no_grad()
    def predict(x):
        return model(x)

`enable_grad()` re-enables tracking inside a `no_grad` region (useful when a
small differentiable sub-step is needed during otherwise grad-free code).
"""
from __future__ import annotations

import functools

# Module-level flag. Read by Function.apply on every op.
_grad_enabled = True


def is_grad_enabled() -> bool:
    """Return whether autograd is currently recording the graph."""
    return _grad_enabled


def set_grad_enabled(mode: bool) -> bool:
    """Set the global grad mode. Returns the previous value."""
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = bool(mode)
    return prev


class _GradMode:
    """Context manager + decorator that forces grad mode on or off."""

    __slots__ = ("_mode", "_prev")

    def __init__(self, mode: bool):
        self._mode = mode
        self._prev = None

    def __enter__(self):
        self._prev = set_grad_enabled(self._mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_grad_enabled(self._prev)
        return False

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _GradMode(self._mode):
                return fn(*args, **kwargs)

        return wrapper


def no_grad() -> _GradMode:
    """Disable autograd inside the `with` block (or decorated function)."""
    return _GradMode(False)


def enable_grad() -> _GradMode:
    """Force autograd on, even inside a surrounding `no_grad` region."""
    return _GradMode(True)
