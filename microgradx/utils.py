"""Utility helpers: activation checkpointing, parameter counting, summaries."""
from __future__ import annotations

from typing import Optional, Tuple

from microgradx.autograd.function import Function
from microgradx.autograd.grad_mode import enable_grad, no_grad


class _Checkpoint(Function):
    """Run a sub-computation without storing its activations, recomputing them
    in the backward pass instead — trading compute for memory."""

    @staticmethod
    def forward(ctx, run_fn, *inputs):
        from microgradx.tensor import Tensor

        ctx.run_fn = run_fn
        ctx.save_for_backward(*inputs)  # raw input arrays
        # Forward without building a graph: no intermediates are retained.
        with no_grad():
            out = run_fn(*[Tensor(a) for a in inputs])
        if isinstance(out, (tuple, list)):
            raise TypeError(
                "checkpoint(fn, ...) requires fn to return a single Tensor"
            )
        return out.data

    @staticmethod
    def backward(ctx, grad_out):
        from microgradx.tensor import Tensor

        # Re-run the forward with grad tracking on to rebuild the local graph,
        # then backprop the upstream gradient through it. Gradients for the
        # checkpointed inputs are returned to the outer graph; gradients for any
        # parameters captured by run_fn accumulate into those real parameters
        # directly (they are leaves of the recomputed graph).
        with enable_grad():
            inputs = [Tensor(a, requires_grad=True) for a in ctx.saved_tensors]
            out = ctx.run_fn(*inputs)
            out.backward(grad_out)
        return (None,) + tuple(t.grad for t in inputs)


def checkpoint(run_fn, *args):
    """Checkpoint ``run_fn(*args)``.

    Equivalent to calling ``run_fn(*args)`` for the forward result, but the
    intermediate activations inside ``run_fn`` are not kept — they are
    recomputed during backward. This cuts peak memory for deep sub-networks
    (e.g. transformer blocks over long sequences) at the cost of one extra
    forward per checkpointed region.

    ``run_fn`` must take Tensors and return a single Tensor. ``args`` are the
    input Tensors to checkpoint; parameters used inside ``run_fn`` still receive
    gradients as usual.

        h = checkpoint(block, x)        # instead of  h = block(x)
    """
    return _Checkpoint.apply(run_fn, *args)


def count_parameters(model) -> int:
    """Total number of scalar elements across ``model.parameters()``."""
    total = 0
    for p in model.parameters():
        total += int(p.data.size)
    return total


def summary(model, input_shape: Optional[Tuple[int, ...]] = None) -> str:
    """Human-readable layer listing with per-parameter shapes and counts.

    Only leaf parameters (via ``named_parameters``) are listed; the total at
    the bottom matches :func:`count_parameters`.

        print(summary(model, input_shape=(1, 784)))
    """
    lines = [model.__class__.__name__]
    if input_shape is not None:
        lines.append(f"  input_shape: {tuple(input_shape)}")
    total = 0
    for name, p in model.named_parameters():
        n = int(p.data.size)
        total += n
        lines.append(f"  {name}: {tuple(p.shape)}  ({n:,} params)")
    # Also list registered buffers (e.g. BatchNorm running stats, Int8 weights)
    bufs = list(model.named_buffers())
    if bufs:
        lines.append("  --- buffers ---")
        for name, b in bufs:
            try:
                shape = tuple(b.shape)
                n = int(b.size)
            except Exception:
                shape = ()
                n = 0
            lines.append(f"  {name}: {shape}  ({n:,} elems)")
    lines.append(f"Total parameters: {total:,}")
    return "\n".join(lines)

