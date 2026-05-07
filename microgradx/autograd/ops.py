"""
Differentiable primitive ops. Each one is a Function with forward + backward.

Broadcasting note: backward returns gradients in the *forward output* shape.
The Tensor.backward driver shrinks them back to input shape via _unbroadcast.
That means each op below can ignore broadcasting.

Math reference (for the non-trivial ones):

  y = a * b               ∂y/∂a = b               ∂y/∂b = a
  y = a / b               ∂y/∂a = 1/b             ∂y/∂b = -a/b²
  y = a @ b               ∂L/∂a = ∂L/∂y @ bᵀ      ∂L/∂b = aᵀ @ ∂L/∂y
  y = xⁿ (scalar n)       ∂y/∂x = n·xⁿ⁻¹
  y = log(x)              ∂y/∂x = 1/x
  y = √x                  ∂y/∂x = 1/(2√x) = 1/(2y)
  y = tanh(x)             ∂y/∂x = 1 - y²
  y = σ(x)                ∂y/∂x = y·(1-y)
  y = softmax(x)_i        ∂yᵢ/∂xⱼ = yᵢ(δᵢⱼ - yⱼ)
                          ⇒ ∂L/∂xⱼ = yⱼ·(gⱼ - Σᵢ gᵢyᵢ)   (g = ∂L/∂y)
  y = log_softmax(x)_i    ∂L/∂xⱼ = gⱼ - softmax(x)ⱼ · Σᵢ gᵢ
"""
from __future__ import annotations
import numpy as np

from microgradx.backend import xp
from microgradx.autograd.function import Function


# ============================================================
# Elementwise binary
# ============================================================

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, g):
        return g, g


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a - b

    @staticmethod
    def backward(ctx, g):
        return g, -g


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        return g * b, g * a


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a / b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        return g / b, -g * a / (b * b)


class Pow(Function):
    @staticmethod
    def forward(ctx, x, exponent):
        ctx.save_for_backward(x)
        ctx.exponent = exponent
        return x ** exponent

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        n = ctx.exponent
        return g * n * (x ** (n - 1)),


# ============================================================
# Elementwise unary
# ============================================================

class Neg(Function):
    @staticmethod
    def forward(ctx, x):
        return -x

    @staticmethod
    def backward(ctx, g):
        return -g,


class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        y = xp.exp(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        return g * y,


class Log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return xp.log(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / x,


class Sqrt(Function):
    @staticmethod
    def forward(ctx, x):
        y = xp.sqrt(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        return g / (2 * y),


class Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        y = xp.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        return g * (1 - y * y),


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        # Numerically stable: σ(x) = 1/(1+exp(-x)), but split on sign
        out = xp.where(x >= 0,
                       1.0 / (1.0 + xp.exp(-x)),
                       xp.exp(x) / (1.0 + xp.exp(x)))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        return g * y * (1 - y),


class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        mask = (x > 0).astype(x.dtype)
        ctx.save_for_backward(mask)
        return x * mask

    @staticmethod
    def backward(ctx, g):
        (mask,) = ctx.saved_tensors
        return g * mask,


class GELU(Function):
    """Approximate GELU (matches the tanh approximation used by GPT)."""
    @staticmethod
    def forward(ctx, x):
        c = xp.sqrt(xp.array(2.0 / xp.pi, dtype=x.dtype))
        u = c * (x + 0.044715 * x ** 3)
        t = xp.tanh(u)
        y = 0.5 * x * (1 + t)
        ctx.save_for_backward(x, t, c)
        return y

    @staticmethod
    def backward(ctx, g):
        x, t, c = ctx.saved_tensors
        # dy/dx = 0.5*(1+t) + 0.5*x*(1-t²) * c * (1 + 3*0.044715*x²)
        dt_dx = c * (1 + 3 * 0.044715 * x * x)
        dy_dx = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dt_dx
        return g * dy_dx,


# ============================================================
# Linear algebra
# ============================================================

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, g):
        a, b = ctx.saved_tensors
        # Use swapaxes so this works for >2D batched matmul too
        return g @ xp.swapaxes(b, -1, -2), xp.swapaxes(a, -1, -2) @ g


# ============================================================
# Reductions
# ============================================================

def _restore_reduced_axes(g, input_shape, axis, keepdims):
    """Re-insert size-1 axes that a reduction collapsed (when keepdims=False)."""
    if keepdims or axis is None:
        return g
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    norm = sorted([(a if a >= 0 else a + len(input_shape)) for a in axes])
    for ax in norm:
        g = xp.expand_dims(g, ax)
    return g


class Sum(Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        ctx.input_shape = x.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        return x.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        g = _restore_reduced_axes(g, ctx.input_shape, ctx.axis, ctx.keepdims)
        return xp.broadcast_to(g, ctx.input_shape).copy(),


class Mean(Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        ctx.input_shape = x.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        if axis is None:
            ctx.n = x.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            ctx.n = int(np.prod([x.shape[a] for a in axes]))
        return x.mean(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx, g):
        g = g / ctx.n
        g = _restore_reduced_axes(g, ctx.input_shape, ctx.axis, ctx.keepdims)
        return xp.broadcast_to(g, ctx.input_shape).copy(),


class Max(Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        if axis is None:
            y = x.max()
            mask = (x == y)
            ctx.save_for_backward(mask.astype(x.dtype))
            ctx.input_shape = x.shape
            ctx.axis = None
            ctx.keepdims = keepdims
            return y if keepdims is False else xp.array(y).reshape((1,) * x.ndim)
        y = x.max(axis=axis, keepdims=True)
        mask = (x == y).astype(x.dtype)
        # Tie-break: divide by count to give equal credit to ties
        mask = mask / mask.sum(axis=axis, keepdims=True)
        ctx.save_for_backward(mask)
        ctx.input_shape = x.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        return y if keepdims else y.squeeze(axis=axis)

    @staticmethod
    def backward(ctx, g):
        (mask,) = ctx.saved_tensors
        g = _restore_reduced_axes(g, ctx.input_shape, ctx.axis, ctx.keepdims)
        return mask * xp.broadcast_to(g, ctx.input_shape),


class Min(Function):
    @staticmethod
    def forward(ctx, x, axis=None, keepdims=False):
        if axis is None:
            y = x.min()
            mask = (x == y).astype(x.dtype)
            ctx.save_for_backward(mask)
            ctx.input_shape = x.shape
            ctx.axis = None
            ctx.keepdims = keepdims
            return y
        y = x.min(axis=axis, keepdims=True)
        mask = (x == y).astype(x.dtype)
        mask = mask / mask.sum(axis=axis, keepdims=True)
        ctx.save_for_backward(mask)
        ctx.input_shape = x.shape
        ctx.axis = axis
        ctx.keepdims = keepdims
        return y if keepdims else y.squeeze(axis=axis)

    @staticmethod
    def backward(ctx, g):
        (mask,) = ctx.saved_tensors
        g = _restore_reduced_axes(g, ctx.input_shape, ctx.axis, ctx.keepdims)
        return mask * xp.broadcast_to(g, ctx.input_shape),


# ============================================================
# Shape ops
# ============================================================

class Reshape(Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.input_shape = x.shape
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, g):
        return g.reshape(ctx.input_shape),


class Transpose(Function):
    @staticmethod
    def forward(ctx, x, axes):
        ctx.axes = axes
        return xp.transpose(x, axes)

    @staticmethod
    def backward(ctx, g):
        # Inverse permutation
        inv = [0] * len(ctx.axes)
        for i, a in enumerate(ctx.axes):
            inv[a] = i
        return xp.transpose(g, inv),


class Expand(Function):
    """Explicit broadcast-to. Backward sums over expanded axes."""
    @staticmethod
    def forward(ctx, x, shape):
        ctx.input_shape = x.shape
        return xp.broadcast_to(x, shape).copy()

    @staticmethod
    def backward(ctx, g):
        # _unbroadcast in Tensor.backward will reduce; we just pass through.
        return g,


class GetItem(Function):
    @staticmethod
    def forward(ctx, x, key):
        ctx.input_shape = x.shape
        ctx.key = key
        return x[key]

    @staticmethod
    def backward(ctx, g):
        out = xp.zeros(ctx.input_shape, dtype=g.dtype)
        # np.add.at handles repeated indices correctly (essential for embedding)
        xp.add.at(out, ctx.key, g)
        return out,


class Concat(Function):
    @staticmethod
    def forward(ctx, *arrays, axis=0):
        ctx.axis = axis
        ctx.sizes = [a.shape[axis] for a in arrays]
        return xp.concatenate(arrays, axis=axis)

    @staticmethod
    def backward(ctx, g):
        splits = xp.split(g, np.cumsum(ctx.sizes)[:-1], axis=ctx.axis)
        return tuple(splits)


# ============================================================
# Softmax / log-softmax (numerically stable)
# ============================================================

class Softmax(Function):
    @staticmethod
    def forward(ctx, x, axis=-1):
        z = x - x.max(axis=axis, keepdims=True)
        e = xp.exp(z)
        y = e / e.sum(axis=axis, keepdims=True)
        ctx.save_for_backward(y)
        ctx.axis = axis
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        # ∂L/∂xⱼ = yⱼ(gⱼ - Σᵢ gᵢyᵢ)
        s = (g * y).sum(axis=ctx.axis, keepdims=True)
        return y * (g - s),


class LogSoftmax(Function):
    @staticmethod
    def forward(ctx, x, axis=-1):
        z = x - x.max(axis=axis, keepdims=True)
        log_sum = xp.log(xp.exp(z).sum(axis=axis, keepdims=True))
        y = z - log_sum
        ctx.save_for_backward(y)
        ctx.axis = axis
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        # softmax(x) = exp(log_softmax(x))
        sm = xp.exp(y)
        s = g.sum(axis=ctx.axis, keepdims=True)
        return g - sm * s,


# ============================================================
# Where / clamp
# ============================================================

class Where(Function):
    @staticmethod
    def forward(ctx, cond, a, b):
        ctx.save_for_backward(cond)
        return xp.where(cond, a, b)

    @staticmethod
    def backward(ctx, g):
        (cond,) = ctx.saved_tensors
        return None, xp.where(cond, g, xp.zeros_like(g)), xp.where(cond, xp.zeros_like(g), g)


class Clamp(Function):
    @staticmethod
    def forward(ctx, x, min=None, max=None):
        ctx.save_for_backward(x)
        ctx.min = min
        ctx.max = max
        return xp.clip(x, min, max)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        mask = xp.ones_like(x)
        if ctx.min is not None:
            mask = mask * (x >= ctx.min)
        if ctx.max is not None:
            mask = mask * (x <= ctx.max)
        return g * mask,
