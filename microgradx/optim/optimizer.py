"""Optimizer base class + gradient clipping helpers."""
from __future__ import annotations
from typing import Iterable, List
import numpy as np

from microgradx.tensor import Tensor


class Optimizer:
    """Holds a list of parameters + state and updates them on `step()`."""

    def __init__(self, params: Iterable[Tensor], defaults: dict):
        self.defaults = defaults
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        for p in params:
            if not isinstance(p, Tensor):
                raise TypeError(f"params must be Tensors, got {type(p)}")
        self.params: List[Tensor] = params
        self.state: List[dict] = [{} for _ in params]
        self.step_count = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


# ---- gradient clipping ----
def clip_grad_norm_(params: Iterable[Tensor], max_norm: float, norm_type: float = 2.0):
    """Rescale gradients in-place so their global Lp-norm is ≤ max_norm.

    Returns the *original* total norm — useful for logging.
    """
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    if norm_type == float("inf"):
        total = float(max(g.abs().max() if hasattr(g, "abs") else np.max(np.abs(g)) for g in grads))
    else:
        total_sq = sum(float((g ** norm_type).sum()) for g in grads)
        total = total_sq ** (1.0 / norm_type)
    clip = max_norm / (total + 1e-6)
    if clip < 1.0:
        for g in grads:
            g *= clip
    return total


def clip_grad_value_(params: Iterable[Tensor], clip_value: float):
    """Element-wise clip every gradient to [-clip_value, clip_value]."""
    for p in params:
        if p.grad is not None:
            np.clip(p.grad, -clip_value, clip_value, out=p.grad)
