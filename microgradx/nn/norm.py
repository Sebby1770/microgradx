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

from microgradx.backend import to_numpy
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


class _BatchNorm(Module):
    """Shared BatchNorm logic. Normalises each channel over the batch (and any
    spatial dims), tracking running statistics for eval.

    Train: normalise with the current batch's mean/var, and fold those into the
    running estimates. Eval: normalise with the running estimates so a single
    sample is treated exactly as it would be at train time. As with LayerNorm,
    the normalisation is built from autograd-aware ops so backprop is automatic;
    the running-stat update is a detached side effect.
    """

    def __init__(self, num_features: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Tensor(np.ones((num_features,), dtype=np.float32),
                                 requires_grad=True)
            self.bias = Tensor(np.zeros((num_features,), dtype=np.float32),
                               requires_grad=True)
        else:
            self.weight = None
            self.bias = None
        # Running stats are buffers (persisted via state_dict, not learnable).
        if track_running_stats:
            self.register_buffer("running_mean",
                                 np.zeros((num_features,), dtype=np.float32))
            self.register_buffer("running_var",
                                 np.ones((num_features,), dtype=np.float32))
        else:
            self.running_mean = None
            self.running_var = None

    def _check_input_dim(self, x: Tensor):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        self._check_input_dim(x)
        # Reduce over the batch axis and every spatial axis — everything but
        # the channel axis (1). Broadcast params/stats to [1, C, 1, ...].
        axes = (0,) + tuple(range(2, x.ndim))
        bshape = [1, self.num_features] + [1] * (x.ndim - 2)

        use_batch = self.training or not self.track_running_stats
        if use_batch:
            mean = x.mean(axis=axes, keepdims=True)
            diff = x - mean
            var = (diff * diff).mean(axis=axes, keepdims=True)
            x_hat = diff / (var + self.eps).sqrt()
            if self.track_running_stats:
                m = to_numpy(mean.data).reshape(-1)
                v = to_numpy(var.data).reshape(-1)
                self.running_mean = ((1 - self.momentum) * self.running_mean
                                     + self.momentum * m).astype(np.float32)
                self.running_var = ((1 - self.momentum) * self.running_var
                                    + self.momentum * v).astype(np.float32)
        else:
            rm = Tensor(self.running_mean.reshape(bshape))
            rv = Tensor(self.running_var.reshape(bshape))
            x_hat = (x - rm) / (rv + self.eps).sqrt()

        if self.affine:
            return x_hat * self.weight.reshape(bshape) + self.bias.reshape(bshape)
        return x_hat


class BatchNorm1d(_BatchNorm):
    """BatchNorm over (N, C) or (N, C, L) inputs."""

    def _check_input_dim(self, x: Tensor):
        if x.ndim not in (2, 3):
            raise ValueError(f"BatchNorm1d expects 2D or 3D input, got {x.ndim}D")


class BatchNorm2d(_BatchNorm):
    """BatchNorm over (N, C, H, W) inputs."""

    def _check_input_dim(self, x: Tensor):
        if x.ndim != 4:
            raise ValueError(f"BatchNorm2d expects 4D input, got {x.ndim}D")


class GroupNorm(Module):
    """Group normalisation over channel groups (Wu & He, 2018).

    Input shape ``(N, C, *)``. Channels are split into ``num_groups`` groups;
    each group is normalised over its channels and all spatial dims, then
    optionally affine-transformed with per-channel ``γ, β``. Independent of
    batch size — works with batch size 1.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = Tensor(
                np.ones((num_channels,), dtype=np.float32), requires_grad=True
            )
            self.bias = Tensor(
                np.zeros((num_channels,), dtype=np.float32), requires_grad=True
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(f"GroupNorm expects at least 2D input, got {x.ndim}D")
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"GroupNorm expected {self.num_channels} channels, got {x.shape[1]}"
            )
        N = x.shape[0]
        C = self.num_channels
        G = self.num_groups
        # Reshape (N, C, *spatial) → (N, G, C//G, *spatial) and normalise
        # over every axis except N and G.
        spatial = x.shape[2:]
        x_g = x.reshape(N, G, C // G, *spatial)
        axes = tuple(range(2, x_g.ndim))
        mean = x_g.mean(axis=axes, keepdims=True)
        diff = x_g - mean
        var = (diff * diff).mean(axis=axes, keepdims=True)
        x_hat = (diff / (var + self.eps).sqrt()).reshape(x.shape)
        if self.affine:
            # Broadcast γ, β over (N, C, 1, 1, ...)
            bshape = [1, C] + [1] * (x.ndim - 2)
            return x_hat * self.weight.reshape(bshape) + self.bias.reshape(bshape)
        return x_hat

    def __repr__(self):
        return (
            f"GroupNorm({self.num_groups}, {self.num_channels}, eps={self.eps})"
        )
