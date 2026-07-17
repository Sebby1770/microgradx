"""Spatial upsampling for NCHW feature maps."""
from __future__ import annotations
from typing import Optional, Tuple, Union

from microgradx.backend import xp
from microgradx.tensor import Tensor
from microgradx.autograd.function import Function
from microgradx.nn.module import Module


def _pair(v) -> Tuple[int, int]:
    if isinstance(v, (tuple, list)):
        if len(v) != 2:
            raise ValueError(f"expected a pair, got {v}")
        return int(v[0]), int(v[1])
    return int(v), int(v)


class _NearestUpsampleFn(Function):
    """Nearest-neighbor upsample of a 4-D NCHW tensor by integer scale factors."""

    @staticmethod
    def forward(ctx, x, scale_h, scale_w):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        sh, sw = int(scale_h), int(scale_w)
        # Repeat along spatial axes: (N,C,H,1,W,1) → expand → reshape
        out = (
            x.reshape(N, C, H, 1, W, 1)
            .repeat(sh, axis=3)
            .repeat(sw, axis=5)
            .reshape(N, C, H * sh, W * sw)
        )
        ctx.scale_h = sh
        ctx.scale_w = sw
        ctx.in_shape = (N, C, H, W)
        return out

    @staticmethod
    def backward(ctx, g):
        # Sum gradients over the repeated blocks back to each input cell.
        N, C, H, W = ctx.in_shape
        sh, sw = ctx.scale_h, ctx.scale_w
        g = g.reshape(N, C, H, sh, W, sw)
        grad_x = g.sum(axis=(3, 5))
        return grad_x, None, None


def interpolate(
    x: Tensor,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    scale_factor: Optional[Union[int, float, Tuple]] = None,
    mode: str = "nearest",
) -> Tensor:
    """Upsample a 4-D NCHW tensor.

    Exactly one of ``size`` or ``scale_factor`` must be given. Currently only
    ``mode="nearest"`` is supported (integer scale factors when using
    ``scale_factor``).
    """
    if x.ndim != 4:
        raise ValueError(f"interpolate expects 4D NCHW input, got {x.ndim}D")
    if mode != "nearest":
        raise ValueError(
            f"only mode='nearest' is supported, got {mode!r}"
        )
    if (size is None) == (scale_factor is None):
        raise ValueError("exactly one of size or scale_factor must be set")

    N, C, H, W = x.shape
    if scale_factor is not None:
        if isinstance(scale_factor, (tuple, list)):
            sh, sw = float(scale_factor[0]), float(scale_factor[1])
        else:
            sh = sw = float(scale_factor)
        if sh != int(sh) or sw != int(sw) or sh < 1 or sw < 1:
            raise ValueError(
                f"nearest upsample requires positive integer scale_factor, got {scale_factor}"
            )
        return _NearestUpsampleFn.apply(x, int(sh), int(sw))

    oh, ow = _pair(size)
    if oh % H != 0 or ow % W != 0:
        raise ValueError(
            f"nearest mode with size=({oh},{ow}) requires integer scale over "
            f"input H={H}, W={W}"
        )
    return _NearestUpsampleFn.apply(x, oh // H, ow // W)


class Upsample(Module):
    """Nearest-neighbor (or size-based) upsample for NCHW tensors.

    Parameters
    ----------
    scale_factor : int or (int, int), optional
        Multiplier for spatial size. Prefer this over ``size`` for integer
        expansion.
    size : int or (int, int), optional
        Target ``(H, W)``. Must be an integer multiple of the input size when
        ``mode="nearest"``.
    mode : str
        Currently only ``"nearest"``.
    """

    def __init__(
        self,
        scale_factor: Optional[Union[int, float, Tuple]] = None,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        mode: str = "nearest",
    ):
        super().__init__()
        if (scale_factor is None) == (size is None):
            raise ValueError("exactly one of scale_factor or size must be set")
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        return interpolate(
            x, size=self.size, scale_factor=self.scale_factor, mode=self.mode
        )

    def __repr__(self):
        if self.scale_factor is not None:
            return f"Upsample(scale_factor={self.scale_factor}, mode={self.mode!r})"
        return f"Upsample(size={self.size}, mode={self.mode!r})"
