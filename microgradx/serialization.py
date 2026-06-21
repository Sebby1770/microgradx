"""Save and load model weights to disk.

Weights are stored as a NumPy ``.npz`` archive: one entry per parameter,
keyed by its dotted name from :meth:`Module.state_dict`, plus a small version
marker. The format is portable, inspectable with ``numpy.load``, and needs no
pickling (``allow_pickle=False``), so loading an Aegis-style untrusted file
cannot execute code.

    mg.save(model, "model.npz")
    model.load_state_dict(mg.load("model.npz"))
    # or the convenience round-trip:
    model.save("model.npz"); model.load("model.npz")
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

_VERSION = 1
_VERSION_KEY = "__microgradx_version__"


def _to_state_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(
        f"save() expects a Module or a state_dict, got {type(obj).__name__}"
    )


def save(obj: Any, path) -> str:
    """Write a Module (or a raw state_dict) to `path` as a ``.npz`` archive.

    Returns the path written.
    """
    sd = _to_state_dict(obj)
    arrays = {k: np.asarray(v) for k, v in sd.items()}
    if _VERSION_KEY in arrays:
        raise ValueError(f"parameter name {_VERSION_KEY!r} is reserved")
    arrays[_VERSION_KEY] = np.array(_VERSION, dtype=np.int64)
    # Pass a file handle so NumPy does not silently append ".npz".
    with open(path, "wb") as f:
        np.savez(f, **arrays)
    return str(path)


def load(path) -> "Dict[str, np.ndarray]":
    """Load a state_dict (name → array) previously written by :func:`save`."""
    with np.load(path, allow_pickle=False) as npz:
        version = int(npz[_VERSION_KEY]) if _VERSION_KEY in npz.files else 0
        if version > _VERSION:
            raise ValueError(
                f"file was written by a newer microgradx (format v{version}); "
                f"this build understands up to v{_VERSION}"
            )
        return {k: npz[k] for k in npz.files if k != _VERSION_KEY}
