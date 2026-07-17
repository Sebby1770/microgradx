"""Evaluation metrics (pure NumPy on ``.data`` — no autograd)."""
from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from microgradx.tensor import Tensor


def accuracy(
    logits,
    targets,
    topk: Union[int, Sequence[int]] = (1,),
) -> Union[float, Tuple[float, ...]]:
    """Top-k classification accuracy.

    Parameters
    ----------
    logits : Tensor or array-like
        Scores of shape ``(N, C)``.
    targets : Tensor or array-like of int
        Class indices of shape ``(N,)``.
    topk : int or sequence of int
        Which top-k accuracies to compute. Default ``(1,)`` (top-1 only).

    Returns
    -------
    float or tuple of float
        Accuracies in ``[0, 1]``. A single float when ``topk`` is a lone int
        or a length-1 sequence; otherwise a tuple matching the order of
        ``topk``.
    """
    if isinstance(logits, Tensor):
        logits = logits.data
    logits = np.asarray(logits)
    if isinstance(targets, Tensor):
        targets = targets.data
    targets = np.asarray(targets).astype(np.int64).reshape(-1)

    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D (N, C), got shape {logits.shape}")
    n, c = logits.shape
    if targets.shape[0] != n:
        raise ValueError(
            f"targets length {targets.shape[0]} != batch size {n}"
        )

    single_int = isinstance(topk, (int, np.integer))
    if single_int:
        ks = (int(topk),)
    else:
        ks = tuple(int(k) for k in topk)
    if not ks:
        raise ValueError("topk must contain at least one k")
    maxk = max(ks)
    if maxk < 1:
        raise ValueError(f"topk values must be >= 1, got {ks}")
    if maxk > c:
        raise ValueError(f"max topk={maxk} exceeds num classes C={c}")

    # Indices of top-maxk predictions per row, descending score order.
    # argpartition then sort within the partition for correct ranking.
    part = np.argpartition(-logits, maxk - 1, axis=1)[:, :maxk]
    row = np.arange(n)[:, None]
    part_scores = logits[row, part]
    order = np.argsort(-part_scores, axis=1)
    top_idx = part[row, order]  # (N, maxk)

    target_col = targets[:, None]  # (N, 1)
    correct = top_idx == target_col  # (N, maxk)

    results = []
    for k in ks:
        # Hit if target appears in first k predictions.
        hit = correct[:, :k].any(axis=1)
        results.append(float(hit.mean()))

    if single_int or len(results) == 1:
        return results[0]
    return tuple(results)
