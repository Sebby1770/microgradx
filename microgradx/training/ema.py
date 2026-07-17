"""Exponential moving average of model parameters."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

import numpy as np

from microgradx.backend import to_numpy


class EMA:
    """Maintain an exponential moving average of a module's parameters.

    Typical usage::

        ema = EMA(model, decay=0.999)
        for batch in loader:
            loss.backward()
            opt.step()
            ema.update()
        with ema.average_parameters():
            evaluate(model)   # uses shadow weights; restores after

    Parameters
    ----------
    model :
        Any object with ``named_parameters()`` yielding ``(name, Tensor)``.
    decay : float
        Smoothing factor in ``[0, 1)``. Higher → slower-moving average.
    """

    def __init__(self, model, decay: float = 0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.model = model
        self.decay = float(decay)
        self.shadow: Dict[str, np.ndarray] = {}
        self.backup: Dict[str, np.ndarray] = {}
        self._n_updates = 0
        for name, p in model.named_parameters():
            self.shadow[name] = to_numpy(p.data).copy()

    def update(self) -> None:
        """Fold current parameters into the shadow average.

        ``shadow = decay * shadow + (1 - decay) * param``
        """
        d = self.decay
        for name, p in self.model.named_parameters():
            if name not in self.shadow:
                self.shadow[name] = to_numpy(p.data).copy()
                continue
            cur = to_numpy(p.data)
            self.shadow[name] = (d * self.shadow[name] + (1.0 - d) * cur).astype(
                self.shadow[name].dtype, copy=False
            )
        self._n_updates += 1

    def apply_shadow(self) -> None:
        """Copy shadow weights into the model (saves originals in ``backup``)."""
        self.backup = {}
        for name, p in self.model.named_parameters():
            self.backup[name] = to_numpy(p.data).copy()
            if name in self.shadow:
                p.data = self.shadow[name].astype(p.data.dtype, copy=True)

    def restore(self) -> None:
        """Restore parameters saved by :meth:`apply_shadow`."""
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data = self.backup[name].astype(p.data.dtype, copy=True)
        self.backup = {}

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        """Temporarily swap in EMA weights; restore on exit.

            with ema.average_parameters():
                evaluate(model)
        """
        self.apply_shadow()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.shadow = {k: np.asarray(v).copy() for k, v in state.items()}

    def __repr__(self) -> str:
        return (
            f"EMA(decay={self.decay}, n_params={len(self.shadow)}, "
            f"updates={self._n_updates})"
        )
