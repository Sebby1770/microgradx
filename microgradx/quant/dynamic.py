"""Dynamic (weight-only) INT8 quantisation for Linear layers.

Workflow::

    from microgradx.quant import quantize_dynamic
    qmodel = quantize_dynamic(model)   # Linear → Int8Linear

Weights are scaled by their absmax (symmetric, zero-point 0) and stored
as int8. At forward time they are dequantised back to float32 and the
matmul runs in fp32 — correct and simple for a learning framework;
production kernels would keep the matmul in int8.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from microgradx.backend import xp
from microgradx.tensor import Tensor
from microgradx.nn.module import Module
from microgradx.nn.linear import Linear


class Observer:
    """Absmax observer for a single weight tensor.

    Records the max absolute value seen (typically once, at quantisation
    time) and exposes a symmetric quantisation scale so that
    ``round(w / scale)`` fits in int8.
    """

    def __init__(self, qmax: int = 127):
        self.qmax = qmax
        self.absmax: float = 0.0

    def observe(self, array) -> None:
        arr = np.asarray(array)
        am = float(np.max(np.abs(arr))) if arr.size else 0.0
        if am > self.absmax:
            self.absmax = am

    def scale(self) -> float:
        if self.absmax == 0.0:
            return 1.0
        return self.absmax / float(self.qmax)

    def quantize(self, array) -> tuple:
        """Return ``(int8_weights, scale)`` for ``array``."""
        self.observe(array)
        s = self.scale()
        arr = np.asarray(array, dtype=np.float32)
        q = np.clip(np.rint(arr / s), -self.qmax - 1, self.qmax).astype(np.int8)
        return q, s


class Int8Linear(Module):
    """Inference Linear with int8 weights + float scale.

    ``forward`` dequantises weights to float32 then does a standard matmul,
    so numerical behaviour is close to the original Linear (within
    quantisation noise). Parameters are frozen (no ``requires_grad``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_int8: np.ndarray,
        scale: float,
        bias: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = float(scale)
        # Buffers so state_dict / save-load still work.
        self.register_buffer("weight_int8", np.asarray(weight_int8, dtype=np.int8))
        if bias is not None:
            self.register_buffer("bias", np.asarray(bias, dtype=np.float32))
        else:
            self.register_buffer("bias", None)

    @classmethod
    def from_linear(cls, linear: Linear) -> "Int8Linear":
        obs = Observer()
        w_q, scale = obs.quantize(linear.weight.numpy())
        b = linear.bias.numpy() if linear.bias is not None else None
        return cls(
            linear.in_features,
            linear.out_features,
            w_q,
            scale,
            bias=b,
        )

    def _dequant_weight(self) -> Tensor:
        w = xp.asarray(self.weight_int8, dtype=xp.float32) * np.float32(self.scale)
        return Tensor(w, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        w = self._dequant_weight()
        out = x @ w.transpose()
        if self.bias is not None:
            out = out + Tensor(self.bias)
        return out

    def __repr__(self):
        return (
            f"Int8Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, scale={self.scale:.6g}, "
            f"bias={self.bias is not None})"
        )


def quantize_dynamic(module: Module) -> Module:
    """Replace every :class:`~microgradx.nn.linear.Linear` with
    :class:`Int8Linear` (weight-only dynamic INT8).

    Returns the same module tree with Linear children swapped out. If
    ``module`` itself is a Linear, returns a new Int8Linear.
    """
    if isinstance(module, Linear):
        return Int8Linear.from_linear(module)

    # Snapshot named modules so we can reassign while walking.
    replacements = []
    for name, child in module.named_modules():
        if name and isinstance(child, Linear):
            replacements.append((name, child))

    for name, lin in replacements:
        parts = name.split(".")
        parent = module
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], Int8Linear.from_linear(lin))

    return module
