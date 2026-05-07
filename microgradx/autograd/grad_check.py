"""
Numerical gradient checking via central differences.

For a scalar function L(x), the central-difference estimate is

    ∂L/∂x_i ≈ (L(x + ε·eᵢ) - L(x - ε·eᵢ)) / (2ε)

We evaluate L at every flat index of every input tensor, then compare the
analytical gradient to the numerical one. Use this to validate any new
Function subclass — if the relative error stays below ~1e-4 for fp32,
your backward is right.
"""
from __future__ import annotations
from typing import Callable, List
import numpy as np

from microgradx.tensor import Tensor


def _scalar_loss(fn, inputs):
    out = fn(*inputs)
    if not isinstance(out, Tensor):
        raise RuntimeError("fn must return a Tensor")
    if out.data.size != 1:
        out = out.sum()
    return float(np.asarray(out.data))


def numerical_grad(
    fn: Callable[..., Tensor],
    inputs: List[Tensor],
    eps: float = 1e-6,
) -> List[np.ndarray]:
    """Central-difference gradient estimate for fn(*inputs) (reduced to scalar)."""
    grads = []
    for t in inputs:
        base = t.data.copy()
        g = np.zeros(base.shape, dtype=np.float64)
        flat_view = base.reshape(-1)
        gflat = g.reshape(-1)
        for i in range(flat_view.size):
            orig = flat_view[i]
            flat_view[i] = orig + eps
            t.data = base.copy()
            plus = _scalar_loss(fn, inputs)
            flat_view[i] = orig - eps
            t.data = base.copy()
            minus = _scalar_loss(fn, inputs)
            gflat[i] = (plus - minus) / (2 * eps)
            flat_view[i] = orig
            t.data = base.copy()
        grads.append(g)
    return grads


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: List[Tensor],
    eps: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    verbose: bool = False,
) -> bool:
    """Compare analytical vs numerical gradients in fp64.

    Casts every input tensor to fp64 for the duration so finite-difference
    noise doesn't masquerade as a bug. Restores originals afterward.
    """
    originals = [(t, t.data.dtype) for t in inputs]
    try:
        for t in inputs:
            t.data = t.data.astype(np.float64)
            t.zero_grad()

        out = fn(*inputs)
        if out.data.size != 1:
            out = out.sum()
        out.backward()
        analytical = [
            t.grad.copy() if t.grad is not None else np.zeros_like(t.data)
            for t in inputs
        ]

        # Reset grads, then numerical
        for t in inputs:
            t.zero_grad()
        numerical = numerical_grad(fn, inputs, eps=eps)

        ok = True
        for i, (a, n) in enumerate(zip(analytical, numerical)):
            diff = np.abs(a - n)
            scale = np.maximum(np.abs(a), np.abs(n))
            # Pass if absolute error is below atol OR relative error below rtol.
            # Same convention as torch.autograd.gradcheck / numpy.allclose.
            passed = bool(((diff < atol) | (diff < rtol * scale)).all())
            if verbose or not passed:
                print(f"  input[{i}]: max abs diff = {diff.max():.2e}, "
                      f"max rel diff = {(diff / np.maximum(scale, 1e-30)).max():.2e}, "
                      f"passed = {passed}")
            ok = ok and passed
        return ok
    finally:
        for t, dtype in originals:
            t.data = t.data.astype(dtype)
            t.zero_grad()
