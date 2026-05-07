"""
LayerNorm, BatchNorm.

LayerNorm math:
    μ = mean(x, last_dims)
    σ² = var(x, last_dims)
    x̂ = (x - μ) / √(σ² + ε)
    y = γ · x̂ + β

Backward could derive analytically, but it's cleaner (and equally fast)
to express LN as a composition of autograd-aware ops and let backprop
do the rest. Same trick PyTorch's pure-Python reference impl uses.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Union

from microgradx.tensor import Tensor
from microgradx.nn.module import Module


class LayerNorm(Module):
    """Normalises across the last `len(normalized_shape)` dims."""

    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]],
                 eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Tensor(np.ones(self.normalized_shape, dtype=np.float32),
                                 requires_grad=True)
            self.bias = Tensor(np.zeros(self.normalized_shape, dtype=np.float32),
                               requires_grad=True)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Last D dims are the feature dims to normalise over.
        D = len(self.normalized_shape)
        axes = tuple(range(x.ndim - D, x.ndim))
        # Use mean/var via sum to stay autograd-aware.
        mean = x.mean(axis=axes, keepdims=True)
        diff = x - mean
        var = (diff * diff).mean(axis=axes, keepdims=True)
        x_hat = diff / (var + self.eps).sqrt()
        if self.elementwise_affine:
            return x_hat * self.weight + self.bias
        return x_hat


class RMSNorm(Module):
    """Root-mean-square norm, used by LLaMA / Mistral.  γ · x / √(mean(x²)+ε)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = Tensor(np.ones((dim,), dtype=np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        rms = ((x * x).mean(axis=-1, keepdims=True) + self.eps).sqrt()
        return (x / rms) * self.weight
