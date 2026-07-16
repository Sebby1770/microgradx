"""
Dropout (inverted) — at train time, drop each activation independently with
probability p, then scale by 1/(1-p) so expected activations are unchanged.
At eval time, this is identity.

Implemented as an explicit Function so it gets a fast custom backward
(elementwise mul by the same mask, no need to remember the upscale factor).
"""
import numpy as np

from microgradx.backend import xp
from microgradx.tensor import Tensor
from microgradx.autograd.function import Function
from microgradx.nn.module import Module


class _DropoutFn(Function):
    @staticmethod
    def forward(ctx, x, p):
        # mask is 1 for keep, scaled by 1/(1-p) so E[output] = input
        keep = 1.0 - p
        mask = (xp.random.rand(*x.shape) < keep).astype(x.dtype) / keep
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, g):
        (mask,) = ctx.saved_tensors
        return g * mask, None


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("dropout p must be in [0, 1)")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        return _DropoutFn.apply(x, self.p)


class _Dropout2dFn(Function):
    """Channel dropout: one Bernoulli mask per (N, C), broadcast over H×W."""

    @staticmethod
    def forward(ctx, x, p):
        # x: (N, C, H, W) — mask shape (N, C, 1, 1)
        N, C = x.shape[0], x.shape[1]
        keep = 1.0 - p
        mask = (xp.random.rand(N, C, 1, 1) < keep).astype(x.dtype) / keep
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, g):
        (mask,) = ctx.saved_tensors
        return g * mask, None


class Dropout2d(Module):
    """Channel dropout for conv feature maps of shape ``(N, C, H, W)``.

    Entire channels are zeroed independently per sample with probability ``p``,
    then surviving channels are scaled by ``1/(1-p)`` (inverted dropout).
    Identity at eval time.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("dropout p must be in [0, 1)")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        if x.ndim != 4:
            raise ValueError(
                f"Dropout2d expects 4D (N,C,H,W) input, got {x.ndim}D"
            )
        return _Dropout2dFn.apply(x, self.p)
