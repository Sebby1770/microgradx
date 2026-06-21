"""
Function and Context — the two pieces every differentiable op is built on.

Design (intentionally close to torch.autograd.Function):

    class MyOp(Function):
        @staticmethod
        def forward(ctx, a, b):
            ctx.save_for_backward(a, b)
            return a * b
        @staticmethod
        def backward(ctx, grad_out):
            a, b = ctx.saved_tensors
            return grad_out * b, grad_out * a

    out = MyOp.apply(x, y)
    out.sum().backward()

`apply` is the only entry point. It:
  1. Strips Tensor wrappers, runs `forward` on raw arrays
  2. Wraps the result in a new Tensor
  3. Records the (fn, ctx, input_tensors) edge on the output for backward
"""
from __future__ import annotations
from typing import Any, Tuple


class Context:
    """Scratch space passed through forward → backward.

    Stores tensors via `save_for_backward` and arbitrary Python state
    via attribute assignment (e.g. `ctx.shape = x.shape`). Keeping these
    separate lets us strip references to large tensors after backward.
    """
    __slots__ = ("saved_tensors", "_attrs", "fn", "input_tensors", "needs_input_grad")

    def __init__(self):
        self.saved_tensors: Tuple = ()
        self._attrs = {}
        self.fn = None
        self.input_tensors = ()
        self.needs_input_grad = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors

    def __setattr__(self, key, value):
        if key in ("saved_tensors", "_attrs", "fn", "input_tensors", "needs_input_grad"):
            object.__setattr__(self, key, value)
        else:
            self._attrs[key] = value

    def __getattr__(self, key):
        try:
            return self._attrs[key]
        except KeyError:
            raise AttributeError(key)


class Function:
    """Subclass and override `forward` and `backward`. Then call `MyFn.apply(...)`."""

    @staticmethod
    def forward(ctx: Context, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx: Context, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        # Local import to avoid a circular dependency at module load time
        from microgradx.tensor import Tensor
        from microgradx.autograd.grad_mode import is_grad_enabled

        ctx = Context()
        ctx.fn = cls
        # Track which positional args were Tensor instances so we can route
        # gradients back to them in the right order.
        input_tensors = tuple(a if isinstance(a, Tensor) else None for a in args)
        ctx.input_tensors = input_tensors
        ctx.needs_input_grad = tuple(
            (t is not None and t.requires_grad) for t in input_tensors
        )

        # Strip Tensor wrappers — forward sees plain arrays / scalars.
        raw_args = tuple(a.data if isinstance(a, Tensor) else a for a in args)
        out_data = cls.forward(ctx, *raw_args, **kwargs)

        # Inside a `no_grad` region we skip graph construction entirely: the
        # output is a leaf, no ctx is attached, and the inputs are not retained.
        requires_grad = any(ctx.needs_input_grad) and is_grad_enabled()
        out = Tensor(out_data, requires_grad=requires_grad)
        if requires_grad:
            out._ctx = ctx
        return out
