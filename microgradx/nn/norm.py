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
from typing import Optional, Tuple, Union

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
                 momentum: Optional[float] = 0.1, affine: bool = True,
                 track_running_stats: bool = True):
        super().__init__()
        if (not isinstance(num_features, (int, np.integer))
                or isinstance(num_features, bool) or num_features <= 0):
            raise ValueError("num_features must be a positive integer")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if momentum is not None and not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1] or None")
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = momentum
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)
        if affine:
            self.weight = Tensor(np.ones((self.num_features,), dtype=np.float32),
                                 requires_grad=True)
            self.bias = Tensor(np.zeros((self.num_features,), dtype=np.float32),
                               requires_grad=True)
        else:
            self.weight = None
            self.bias = None
        # Running stats are buffers (persisted via state_dict, not learnable).
        if track_running_stats:
            self.register_buffer("running_mean",
                                 np.zeros((self.num_features,), dtype=np.float32))
            self.register_buffer("running_var",
                                 np.ones((self.num_features,), dtype=np.float32))
            self.register_buffer("num_batches_tracked",
                                 np.array(0, dtype=np.int64))
            # v0.2-v0.6 BatchNorm checkpoints predate this counter. Loading
            # one retains strict validation for every other key while
            # initializing the new state deterministically.
            object.__setattr__(
                self,
                "_state_load_defaults",
                {"num_batches_tracked": np.array(0, dtype=np.int64)},
            )
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None

    def reset_running_stats(self) -> None:
        """Reset running statistics without replacing their buffer objects."""
        if self.track_running_stats:
            self.running_mean.fill(0)
            self.running_var.fill(1)
            self.num_batches_tracked.fill(0)

    def reset_parameters(self) -> None:
        """Restore affine parameters and running statistics to their defaults."""
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill(1)
            self.bias.data.fill(0)

    def _check_input_dim(self, x: Tensor):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        self._check_input_dim(x)
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"expected {self.num_features} channels at dimension 1, "
                f"got {x.shape[1]} for input shape {x.shape}"
            )
        if x.dtype.kind != "f":
            raise TypeError(
                "BatchNorm expects real floating-point input, "
                f"got dtype {x.dtype}"
            )
        # Reduce over the batch axis and every spatial axis — everything but
        # the channel axis (1). Broadcast params/stats to [1, C, 1, ...].
        axes = (0,) + tuple(range(2, x.ndim))
        bshape = [1, self.num_features] + [1] * (x.ndim - 2)

        use_batch = self.training or not self.track_running_stats
        if use_batch:
            samples_per_channel = int(np.prod([x.shape[a] for a in axes]))
            if samples_per_channel <= 1:
                raise ValueError(
                    "expected more than one value per channel when using "
                    f"batch statistics, got input shape {x.shape}"
                )
            mean = x.mean(axis=axes, keepdims=True)
            diff = x - mean
            # The normalization itself uses the biased (population) variance,
            # matching PyTorch and the original BatchNorm paper.
            var = (diff * diff).mean(axis=axes, keepdims=True)
            x_hat = diff / (var + self.eps).sqrt()
            if self.track_running_stats:
                m = to_numpy(mean.data).reshape(-1)
                v = to_numpy(var.data).reshape(-1)
                self.num_batches_tracked[...] += 1
                if self.momentum is None:
                    factor = 1.0 / float(self.num_batches_tracked.item())
                else:
                    factor = self.momentum
                # Running variance stores the unbiased sample estimate even
                # though the forward pass normalizes with the biased estimate.
                unbiased_v = v * samples_per_channel / (samples_per_channel - 1)
                self.running_mean[...] = (
                    (1 - factor) * self.running_mean + factor * m
                ).astype(self.running_mean.dtype, copy=False)
                self.running_var[...] = (
                    (1 - factor) * self.running_var + factor * unbiased_v
                ).astype(self.running_var.dtype, copy=False)
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
