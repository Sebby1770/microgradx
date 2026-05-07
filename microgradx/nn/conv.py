"""
Conv2d implemented via im2col → matmul → reshape.

Why im2col?
  Naive 6-loop convolution is correct but ~30× slower than NumPy GEMM.
  im2col rearranges every receptive field into a column of a single 2D
  matrix. Then the convolution becomes:

        Y_flat (N*OH*OW, C_out) = cols (N*OH*OW, C_in*KH*KW) @ Wᵀ (...)

  We pay one extra reshape but get NumPy's BLAS GEMM for the heavy lifting.

Backward:
  dY → d(cols) → col2im (scatter-add of overlapping patches) → dX
  dY → cols.T @ dY → reshape → dW
"""
from __future__ import annotations
import math
import numpy as np

from microgradx.backend import xp
from microgradx.tensor import Tensor
from microgradx.autograd.function import Function
from microgradx.nn.module import Module
from microgradx.nn import init


def _im2col(x, kh, kw, stride, padding):
    """x: (N, C, H, W) → cols: (N, OH, OW, C, KH, KW)."""
    N, C, H, W = x.shape
    sh, sw = stride
    ph, pw = padding
    OH = (H + 2 * ph - kh) // sh + 1
    OW = (W + 2 * pw - kw) // sw + 1
    # Pad with zeros (constant)
    if ph or pw:
        xp_pad = xp.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    else:
        xp_pad = x
    cols = xp.zeros((N, C, kh, kw, OH, OW), dtype=x.dtype)
    for i in range(kh):
        i_max = i + sh * OH
        for j in range(kw):
            j_max = j + sw * OW
            cols[:, :, i, j, :, :] = xp_pad[:, :, i:i_max:sh, j:j_max:sw]
    # Reorder: (N, OH, OW, C, KH, KW)
    cols = cols.transpose(0, 4, 5, 1, 2, 3)
    return cols, (OH, OW)


def _col2im(cols, x_shape, kh, kw, stride, padding):
    """Inverse of _im2col, summing overlapping patches (correct for backward)."""
    N, C, H, W = x_shape
    sh, sw = stride
    ph, pw = padding
    OH = (H + 2 * ph - kh) // sh + 1
    OW = (W + 2 * pw - kw) // sw + 1
    cols = cols.transpose(0, 3, 4, 5, 1, 2)  # back to (N, C, KH, KW, OH, OW)
    Hp, Wp = H + 2 * ph, W + 2 * pw
    out = xp.zeros((N, C, Hp, Wp), dtype=cols.dtype)
    for i in range(kh):
        i_max = i + sh * OH
        for j in range(kw):
            j_max = j + sw * OW
            out[:, :, i:i_max:sh, j:j_max:sw] += cols[:, :, i, j, :, :]
    return out[:, :, ph:ph + H, pw:pw + W]


class _Conv2dFn(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        # x: (N, Cin, H, W); weight: (Cout, Cin, KH, KW); bias: (Cout,) or None
        N, Cin, H, W = x.shape
        Cout, _, KH, KW = weight.shape
        cols, (OH, OW) = _im2col(x, KH, KW, stride, padding)
        # cols: (N, OH, OW, Cin, KH, KW) → 2D
        cols_flat = cols.reshape(N * OH * OW, Cin * KH * KW)
        W_flat = weight.reshape(Cout, Cin * KH * KW)
        out = cols_flat @ W_flat.T  # (N*OH*OW, Cout)
        out = out.reshape(N, OH, OW, Cout).transpose(0, 3, 1, 2)
        if bias is not None:
            out = out + bias.reshape(1, Cout, 1, 1)
        ctx.save_for_backward(cols_flat, W_flat)
        ctx.x_shape = x.shape
        ctx.w_shape = weight.shape
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.OH, ctx.OW = OH, OW
        return out

    @staticmethod
    def backward(ctx, g):
        cols_flat, W_flat = ctx.saved_tensors
        N, Cin, H, W = ctx.x_shape
        Cout, _, KH, KW = ctx.w_shape
        OH, OW = ctx.OH, ctx.OW
        # Reshape upstream grad: (N, Cout, OH, OW) → (N*OH*OW, Cout)
        g_flat = g.transpose(0, 2, 3, 1).reshape(N * OH * OW, Cout)
        dW_flat = g_flat.T @ cols_flat                          # (Cout, Cin*KH*KW)
        dcols_flat = g_flat @ W_flat                            # (N*OH*OW, Cin*KH*KW)
        dW = dW_flat.reshape(Cout, Cin, KH, KW)
        # col2im for input grad
        dcols = dcols_flat.reshape(N, OH, OW, Cin, KH, KW)
        dX = _col2im(dcols, ctx.x_shape, KH, KW, ctx.stride, ctx.padding)
        db = g.sum(axis=(0, 2, 3)) if ctx.has_bias else None
        return dX, dW, db, None, None


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = Tensor(
            np.zeros((out_channels, in_channels, *kernel_size), dtype=np.float32),
            requires_grad=True,
        )
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_channels * kernel_size[0] * kernel_size[1]
            bound = 1.0 / math.sqrt(fan_in)
            self.bias = Tensor(
                np.random.uniform(-bound, bound, (out_channels,)).astype(np.float32),
                requires_grad=True,
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return _Conv2dFn.apply(x, self.weight, self.bias, self.stride, self.padding)

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding})")


# ---- Pooling helpers (max + avg) ----
class _MaxPool2dFn(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride):
        N, C, H, W = x.shape
        kh, kw = kernel_size
        sh, sw = stride
        OH = (H - kh) // sh + 1
        OW = (W - kw) // sw + 1
        # Build patches via im2col with no padding
        cols, _ = _im2col(x, kh, kw, (sh, sw), (0, 0))  # (N, OH, OW, C, KH, KW)
        cols = cols.transpose(0, 3, 1, 2, 4, 5).reshape(N, C, OH, OW, kh * kw)
        idx = xp.argmax(cols, axis=-1)
        # Gather max
        out = xp.take_along_axis(cols, idx[..., None], axis=-1).squeeze(-1)
        ctx.save_for_backward(idx)
        ctx.x_shape = x.shape
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.OH, ctx.OW = OH, OW
        return out

    @staticmethod
    def backward(ctx, g):
        N, C, H, W = ctx.x_shape
        kh, kw = ctx.kernel_size
        sh, sw = ctx.stride
        OH, OW = ctx.OH, ctx.OW
        (idx,) = ctx.saved_tensors
        # Scatter g into the position of the max within each patch
        dcols = xp.zeros((N, C, OH, OW, kh * kw), dtype=g.dtype)
        xp.put_along_axis(dcols, idx[..., None], g[..., None], axis=-1)
        dcols = dcols.reshape(N, C, OH, OW, kh, kw).transpose(0, 2, 3, 1, 4, 5)
        return _col2im(dcols, ctx.x_shape, kh, kw, ctx.stride, (0, 0)), None, None


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return _MaxPool2dFn.apply(x, self.kernel_size, self.stride)


class Flatten(Module):
    """Flatten everything except the batch dim."""
    def forward(self, x):
        return x.reshape(x.shape[0], -1)
