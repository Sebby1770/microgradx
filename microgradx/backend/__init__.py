"""
Backend abstraction. Defaults to NumPy. If CuPy is available and
MICROGRADX_BACKEND=cupy is set, the GPU backend is used transparently.

Every other module imports `xp` from here and writes purely against the
NumPy/CuPy array API surface.
"""
from __future__ import annotations
import os

_BACKEND_NAME = os.environ.get("MICROGRADX_BACKEND", "numpy").lower()

if _BACKEND_NAME == "cupy":
    try:
        import cupy as xp  # type: ignore
        DEVICE = "cuda"
    except ImportError:
        import numpy as xp
        DEVICE = "cpu"
        _BACKEND_NAME = "numpy"
else:
    import numpy as xp
    DEVICE = "cpu"


def is_gpu() -> bool:
    return DEVICE == "cuda"


def to_numpy(arr):
    """Bring an array back to NumPy regardless of backend."""
    if _BACKEND_NAME == "cupy":
        return xp.asnumpy(arr)
    return arr


def asarray(data, dtype=None):
    """Backend-agnostic array constructor."""
    return xp.asarray(data, dtype=dtype)


__all__ = ["xp", "DEVICE", "is_gpu", "to_numpy", "asarray"]
