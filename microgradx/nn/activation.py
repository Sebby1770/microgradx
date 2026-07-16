"""Activation functions as nn Modules (most just delegate to ops)."""
from microgradx.backend import xp
from microgradx.tensor import Tensor
from microgradx.autograd.function import Function
from microgradx.nn.module import Module
from microgradx.autograd.ops import GELU as _GELUFn


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return _GELUFn.apply(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.axis)


class LeakyReLU(Module):
    """Leaky ReLU: ``max(0, x) + negative_slope * min(0, x)``.

    Implemented as ``slope * x + (1 - slope) * relu(x)`` so it reuses the
    existing ReLU autograd op.
    """

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        s = self.negative_slope
        return x * s + x.relu() * (1.0 - s)


class SiLU(Module):
    """Sigmoid Linear Unit (Swish): ``x * sigmoid(x)``."""

    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


class _SoftplusFn(Function):
    """Numerically stable softplus: ``log(1 + exp(x))`` via ``logaddexp(0, x)``.

    Gradient is ``sigmoid(x)``.
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return xp.logaddexp(0.0, x).astype(x.dtype, copy=False)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        # sigmoid(x) = 1 / (1 + exp(-x)), clipped for stability
        sig = 1.0 / (1.0 + xp.exp(-xp.clip(x, -60.0, 60.0)))
        return g * sig.astype(g.dtype, copy=False),


class Softplus(Module):
    """Softplus: ``log(1 + exp(x))`` (smooth ReLU)."""

    def forward(self, x: Tensor) -> Tensor:
        return _SoftplusFn.apply(x)
