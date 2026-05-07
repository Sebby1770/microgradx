"""
Loss functions.

CrossEntropy is implemented as a fused log_softmax + NLL so we get a
numerically stable gradient *and* a clean closed-form ∂L/∂logits = softmax-target.
"""
from __future__ import annotations
import numpy as np

from microgradx.tensor import Tensor
from microgradx.backend import xp
from microgradx.autograd.function import Function
from microgradx.nn.module import Module


class _CrossEntropyFn(Function):
    """Combined log-softmax + NLL with int targets.

    Forward expects logits (N, C) and target int (N,).
    Loss = mean over N of  -log_softmax(logits)[n, target[n]]
    Gradient w.r.t. logits = (softmax(logits) - one_hot(target)) / N
    """
    @staticmethod
    def forward(ctx, logits, target, reduction="mean"):
        N, C = logits.shape
        z = logits - logits.max(axis=-1, keepdims=True)
        ez = xp.exp(z)
        log_sum = xp.log(ez.sum(axis=-1, keepdims=True))
        log_softmax = z - log_sum                        # (N, C)
        nll = -log_softmax[xp.arange(N), target]         # (N,)
        if reduction == "mean":
            loss = nll.mean()
            scale = 1.0 / N
        elif reduction == "sum":
            loss = nll.sum()
            scale = 1.0
        else:  # "none"
            loss = nll
            scale = 1.0
        # Save softmax for backward (= exp of log_softmax)
        sm = xp.exp(log_softmax)
        ctx.save_for_backward(sm)
        ctx.target = target
        ctx.reduction = reduction
        ctx.scale = scale
        return loss

    @staticmethod
    def backward(ctx, g):
        (sm,) = ctx.saved_tensors
        N, C = sm.shape
        target = ctx.target
        # one-hot grad: (sm - one_hot) * upstream * scale
        grad_logits = sm.copy()
        grad_logits[xp.arange(N), target] -= 1.0
        if ctx.reduction == "none":
            # Per-sample gradient
            grad_logits = grad_logits * g[:, None]
        else:
            grad_logits = grad_logits * (g * ctx.scale)
        return grad_logits, None, None


def cross_entropy(logits: Tensor, target, reduction: str = "mean") -> Tensor:
    if isinstance(target, Tensor):
        target = target.data.astype(np.int64)
    else:
        target = np.asarray(target, dtype=np.int64)
    return _CrossEntropyFn.apply(logits, target, reduction)


class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, target) -> Tensor:
        return cross_entropy(logits, target, self.reduction)


def mse_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    if not isinstance(target, Tensor):
        target = Tensor(target)
    diff = pred - target
    sq = diff * diff
    if reduction == "mean":
        return sq.mean()
    if reduction == "sum":
        return sq.sum()
    return sq


class MSELoss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target) -> Tensor:
        return mse_loss(pred, target, self.reduction)
