"""Spatial upsampling for NCHW feature maps (nearest + bilinear)."""
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


class _BilinearUpsampleFn(Function):
    """Bilinear upsample of a 4-D NCHW tensor by integer scale factors.

    Output pixel ``(i, j)`` samples the continuous input coordinate
    ``(i / sh, j / sw)`` with align-corners=False-style mapping that lands
    integer multiples exactly on input pixels (``oh=0 → 0``, ``oh=sh → 1``, …).
    """

    @staticmethod
    def forward(ctx, x, scale_h, scale_w):
        N, C, H, W = x.shape
        sh, sw = int(scale_h), int(scale_w)
        out_h, out_w = H * sh, W * sw

        # Source coordinates in input space for each output row/col.
        # oh / sh  →  0, 1/sh, 2/sh, ..., (H*sh-1)/sh
        ys = xp.arange(out_h, dtype=x.dtype) / sh
        xs = xp.arange(out_w, dtype=x.dtype) / sw

        y0 = xp.floor(ys).astype(xp.int64)
        x0 = xp.floor(xs).astype(xp.int64)
        y1 = xp.minimum(y0 + 1, H - 1)
        x1 = xp.minimum(x0 + 1, W - 1)
        wy = ys - y0.astype(x.dtype)
        wx = xs - x0.astype(x.dtype)

        # Gather four corners: shapes broadcast to (N, C, out_h, out_w)
        # x[:, :, y0, :] → need advanced indexing
        # Build (out_h, out_w) weight maps and gather.
        # corners: Ia = x[..., y0, x0], Ib = x[..., y0, x1],
        #          Ic = x[..., y1, x0], Id = x[..., y1, x1]
        # Use mesh for indexing
        # y0: (out_h,), x0: (out_w,)
        # Index: x[:, :, y0[:, None], x0[None, :]]
        Ia = x[:, :, y0[:, None], x0[None, :]]
        Ib = x[:, :, y0[:, None], x1[None, :]]
        Ic = x[:, :, y1[:, None], x0[None, :]]
        Id = x[:, :, y1[:, None], x1[None, :]]

        wa = (1.0 - wy)[:, None] * (1.0 - wx)[None, :]  # (out_h, out_w)
        wb = (1.0 - wy)[:, None] * wx[None, :]
        wc = wy[:, None] * (1.0 - wx)[None, :]
        wd = wy[:, None] * wx[None, :]

        out = (
            Ia * wa[None, None, :, :]
            + Ib * wb[None, None, :, :]
            + Ic * wc[None, None, :, :]
            + Id * wd[None, None, :, :]
        )

        ctx.scale_h = sh
        ctx.scale_w = sw
        ctx.in_shape = (N, C, H, W)
        ctx.y0, ctx.y1 = y0, y1
        ctx.x0, ctx.x1 = x0, x1
        ctx.wa, ctx.wb, ctx.wc, ctx.wd = wa, wb, wc, wd
        return out

    @staticmethod
    def backward(ctx, g):
        N, C, H, W = ctx.in_shape
        y0, y1 = ctx.y0, ctx.y1
        x0, x1 = ctx.x0, ctx.x1
        wa, wb, wc, wd = ctx.wa, ctx.wb, ctx.wc, ctx.wd
        out_h, out_w = g.shape[2], g.shape[3]

        grad_x = xp.zeros((N, C, H, W), dtype=g.dtype)
        # Flatten batch×channel so we can use add.at on 2-D spatial maps.
        gx = grad_x.reshape(N * C, H, W)
        gg = g.reshape(N * C, out_h, out_w)

        # Broadcast source index grids to (out_h, out_w).
        yy0 = xp.broadcast_to(y0[:, None], (out_h, out_w))
        yy1 = xp.broadcast_to(y1[:, None], (out_h, out_w))
        xx0 = xp.broadcast_to(x0[None, :], (out_h, out_w))
        xx1 = xp.broadcast_to(x1[None, :], (out_h, out_w))

        def _scatter(ys, xs, weight):
            contrib = gg * weight[None, :, :]  # (NC, oh, ow)
            for nc in range(N * C):
                xp.add.at(gx[nc], (ys, xs), contrib[nc])

        _scatter(yy0, xx0, wa)
        _scatter(yy0, xx1, wb)
        _scatter(yy1, xx0, wc)
        _scatter(yy1, xx1, wd)
        return grad_x, None, None


def interpolate(
    x: Tensor,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    scale_factor: Optional[Union[int, float, Tuple]] = None,
    mode: str = "nearest",
) -> Tensor:
    """Upsample a 4-D NCHW tensor.

    Exactly one of ``size`` or ``scale_factor`` must be given.
    Supported modes: ``"nearest"``, ``"bilinear"`` (integer scale factors).
    """
    if x.ndim != 4:
        raise ValueError(f"interpolate expects 4D NCHW input, got {x.ndim}D")
    if mode not in ("nearest", "bilinear"):
        raise ValueError(
            f"mode must be 'nearest' or 'bilinear', got {mode!r}"
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
                f"upsample requires positive integer scale_factor, got {scale_factor}"
            )
        sh, sw = int(sh), int(sw)
        if mode == "nearest":
            return _NearestUpsampleFn.apply(x, sh, sw)
        return _BilinearUpsampleFn.apply(x, sh, sw)

    oh, ow = _pair(size)
    if oh % H != 0 or ow % W != 0:
        raise ValueError(
            f"{mode} mode with size=({oh},{ow}) requires integer scale over "
            f"input H={H}, W={W}"
        )
    sh, sw = oh // H, ow // W
    if mode == "nearest":
        return _NearestUpsampleFn.apply(x, sh, sw)
    return _BilinearUpsampleFn.apply(x, sh, sw)


class Upsample(Module):
    """Upsample for NCHW tensors (nearest or bilinear).

    Parameters
    ----------
    scale_factor : int or (int, int), optional
        Multiplier for spatial size. Prefer this over ``size`` for integer
        expansion.
    size : int or (int, int), optional
        Target ``(H, W)``. Must be an integer multiple of the input size.
    mode : {"nearest", "bilinear"}
        Interpolation mode. ``"bilinear"`` requires integer scale factors.
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
        if mode not in ("nearest", "bilinear"):
            raise ValueError(
                f"mode must be 'nearest' or 'bilinear', got {mode!r}"
            )
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
