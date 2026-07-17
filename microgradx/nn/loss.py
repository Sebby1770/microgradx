"""
Loss functions.

CrossEntropy is implemented as a fused log_softmax + NLL so we get a
numerically stable gradient *and* a clean closed-form ∂L/∂logits = softmax-target.

BCEWithLogits uses the numerically stable formulation
  max(x, 0) - x·y + log(1 + exp(-|x|))
so large |logits| never overflow.
"""
from __future__ import annotations
import numpy as np

from microgradx.tensor import Tensor
from microgradx.backend import xp
from microgradx.autograd.function import Function
from microgradx.nn.module import Module


class _CrossEntropyFn(Function):
    """Combined log-softmax + NLL with int targets and optional label smoothing.

    Forward expects logits (N, C) and target int (N,).
    With label_smoothing=0:
      Loss = mean over N of  -log_softmax(logits)[n, target[n]]
      Gradient w.r.t. logits = (softmax(logits) - one_hot(target)) / N
    With label_smoothing=α > 0 (PyTorch-style soft targets):
      y_k = (1-α) on the true class, α/C elsewhere  →  y = (1-α)·one_hot + α/C
      Loss = -sum_k y_k log_softmax_k
      Grad  = (softmax - y) / N
    """
    @staticmethod
    def forward(ctx, logits, target, reduction="mean", label_smoothing=0.0):
        N, C = logits.shape
        z = logits - logits.max(axis=-1, keepdims=True)
        ez = xp.exp(z)
        log_sum = xp.log(ez.sum(axis=-1, keepdims=True))
        log_softmax = z - log_sum                        # (N, C)
        sm = xp.exp(log_softmax)
        ls = float(label_smoothing)

        if ls > 0.0:
            # Soft targets: (1-α) on true class + α/C everywhere
            nll_hard = -log_softmax[xp.arange(N), target]  # (N,)
            # mean over classes of -log_softmax
            smooth_loss = -log_softmax.mean(axis=-1)       # (N,)
            nll = (1.0 - ls) * nll_hard + ls * smooth_loss
            # soft one-hot for backward
            soft = xp.full_like(sm, ls / C)
            soft[xp.arange(N), target] = 1.0 - ls + ls / C
            target_dist = soft
        else:
            nll = -log_softmax[xp.arange(N), target]       # (N,)
            target_dist = None

        if reduction == "mean":
            loss = nll.mean()
            scale = 1.0 / N
        elif reduction == "sum":
            loss = nll.sum()
            scale = 1.0
        else:  # "none"
            loss = nll
            scale = 1.0

        ctx.save_for_backward(sm)
        ctx.target = target
        ctx.target_dist = target_dist
        ctx.reduction = reduction
        ctx.scale = scale
        ctx.label_smoothing = ls
        return loss

    @staticmethod
    def backward(ctx, g):
        (sm,) = ctx.saved_tensors
        N, C = sm.shape
        if ctx.target_dist is not None:
            grad_logits = sm - ctx.target_dist
        else:
            grad_logits = sm.copy()
            grad_logits[xp.arange(N), ctx.target] -= 1.0
        if ctx.reduction == "none":
            grad_logits = grad_logits * g[:, None]
        else:
            grad_logits = grad_logits * (g * ctx.scale)
        return grad_logits, None, None, None


def cross_entropy(
    logits: Tensor,
    target,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Tensor:
    """Negative log-likelihood of a softmax classifier with optional label smoothing.

    Parameters
    ----------
    logits : Tensor
        Unnormalised scores of shape ``(N, C)``.
    target : array-like of int
        Class indices of shape ``(N,)``.
    reduction : {"mean", "sum", "none"}
    label_smoothing : float
        If > 0, soft targets ``y = (1-α)·one_hot + α/C`` are used.
    """
    if label_smoothing < 0.0 or label_smoothing > 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1], got {label_smoothing}")
    if isinstance(target, Tensor):
        target = target.data.astype(np.int64)
    else:
        target = np.asarray(target, dtype=np.int64)
    return _CrossEntropyFn.apply(logits, target, reduction, float(label_smoothing))


class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = "mean", label_smoothing: float = 0.0):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: Tensor, target) -> Tensor:
        return cross_entropy(
            logits, target, self.reduction, label_smoothing=self.label_smoothing
        )


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


# ---------------------------------------------------------------------------
# Binary cross-entropy
# ---------------------------------------------------------------------------

class _BCEWithLogitsFn(Function):
    """Stable binary cross-entropy with logits.

    Per-element loss:
        max(x, 0) - x·y + log(1 + exp(-|x|))

    Gradient w.r.t. logits:  sigmoid(x) - y
    """

    @staticmethod
    def forward(ctx, logits, target, reduction="mean"):
        x = logits
        y = target
        # Stable per-element BCE
        # max(x,0) - x*y + log(1+exp(-|x|))
        max_val = xp.maximum(x, 0.0)
        loss_el = max_val - x * y + xp.log1p(xp.exp(-xp.abs(x)))
        n = float(loss_el.size)
        if reduction == "mean":
            loss = loss_el.mean()
            scale = 1.0 / n
        elif reduction == "sum":
            loss = loss_el.sum()
            scale = 1.0
        else:
            loss = loss_el
            scale = 1.0
        # sigmoid for backward: 1 / (1 + exp(-x)), stable
        sig = 1.0 / (1.0 + xp.exp(-xp.clip(x, -60.0, 60.0)))
        ctx.save_for_backward(sig, y)
        ctx.reduction = reduction
        ctx.scale = scale
        return loss

    @staticmethod
    def backward(ctx, g):
        sig, y = ctx.saved_tensors
        grad = sig - y
        if ctx.reduction == "none":
            grad = grad * g
        else:
            grad = grad * (g * ctx.scale)
        # target typically does not need grad
        return grad, None, None


def binary_cross_entropy_with_logits(
    logits: Tensor, target, reduction: str = "mean"
) -> Tensor:
    """Binary cross-entropy between logits and targets in ``[0, 1]``.

    Uses the numerically stable formulation
    ``max(x,0) - x·y + log(1+exp(-|x|))``.
    """
    if not isinstance(target, Tensor):
        target = Tensor(target)
    if logits.shape != target.shape:
        raise ValueError(
            f"logits shape {logits.shape} != target shape {target.shape}"
        )
    return _BCEWithLogitsFn.apply(logits, target, reduction)


class BCEWithLogitsLoss(Module):
    """Stable BCE that accepts raw logits (no sigmoid needed)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, target) -> Tensor:
        return binary_cross_entropy_with_logits(logits, target, self.reduction)


class _BCEFn(Function):
    """BCE between probabilities ``p`` and targets ``y`` in ``[0, 1]``.

    ``-(y log p + (1-y) log(1-p))`` with inputs clamped away from {0,1}.
    """

    @staticmethod
    def forward(ctx, pred, target, reduction="mean", eps=1e-7):
        p = xp.clip(pred, eps, 1.0 - eps)
        y = target
        loss_el = -(y * xp.log(p) + (1.0 - y) * xp.log(1.0 - p))
        n = float(loss_el.size)
        if reduction == "mean":
            loss = loss_el.mean()
            scale = 1.0 / n
        elif reduction == "sum":
            loss = loss_el.sum()
            scale = 1.0
        else:
            loss = loss_el
            scale = 1.0
        ctx.save_for_backward(p, y)
        ctx.reduction = reduction
        ctx.scale = scale
        ctx.eps = eps
        return loss

    @staticmethod
    def backward(ctx, g):
        p, y = ctx.saved_tensors
        # d/dp [-(y log p + (1-y) log(1-p))] = -y/p + (1-y)/(1-p)
        grad = -y / p + (1.0 - y) / (1.0 - p)
        if ctx.reduction == "none":
            grad = grad * g
        else:
            grad = grad * (g * ctx.scale)
        return grad, None, None, None


def binary_cross_entropy(
    pred: Tensor, target, reduction: str = "mean", eps: float = 1e-7
) -> Tensor:
    """Binary cross-entropy between probabilities and targets in ``[0, 1]``.

    Prefer :func:`binary_cross_entropy_with_logits` when you have logits.
    """
    if not isinstance(target, Tensor):
        target = Tensor(target)
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {pred.shape} != target shape {target.shape}"
        )
    return _BCEFn.apply(pred, target, reduction, eps)


class BCELoss(Module):
    """BCE for probability inputs (clamped for log stability)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: Tensor, target) -> Tensor:
        return binary_cross_entropy(pred, target, self.reduction)
