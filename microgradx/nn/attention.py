"""
Multi-head scaled dot-product attention.

Math (per head):
  Q, K, V = X·W_q, X·W_k, X·W_v        each (B, T, d_head)
  S = Q·Kᵀ / √d_head                    (B, T, T)
  P = softmax(S + mask)                 (B, T, T)
  O = P·V                                (B, T, d_head)

Multi-head: split d_model into n_heads groups, run in parallel via batched
matmul, concat back. Implemented with primitive ops so autograd handles it.
"""
from __future__ import annotations
import math
import numpy as np

from microgradx.tensor import Tensor
from microgradx.backend import xp
from microgradx.nn.module import Module
from microgradx.nn.linear import Linear
from microgradx.nn.dropout import Dropout


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None, dropout_p: float = 0.0
) -> Tensor:
    """q, k, v shapes: (..., T_q, d), (..., T_k, d), (..., T_k, d_v)."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(*range(q.ndim - 2), -1, -2) / math.sqrt(d_k)
    if mask is not None:
        # Add a large negative number where mask is 0 (or True if bool mask)
        if isinstance(mask, Tensor):
            mask_arr = mask.data
        else:
            mask_arr = np.asarray(mask)
        bias = np.where(mask_arr, 0.0, -1e9).astype(scores.data.dtype)
        scores = scores + Tensor(bias)
    attn = scores.softmax(axis=-1)
    if dropout_p > 0:
        from microgradx.nn.dropout import _DropoutFn
        attn = _DropoutFn.apply(attn, dropout_p)
    return attn @ v


class MultiHeadAttention(Module):
    """Causal mask is optional via the `causal=True` flag at forward time."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = Linear(d_model, d_model, bias=bias)
        self.k_proj = Linear(d_model, d_model, bias=bias)
        self.v_proj = Linear(d_model, d_model, bias=bias)
        self.out_proj = Linear(d_model, d_model, bias=bias)
        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)
        self.dropout_p = dropout

    def _split_heads(self, x: Tensor) -> Tensor:
        # (B, T, D) -> (B, T, H, D/H) -> (B, H, T, D/H)
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor) -> Tensor:
        # (B, H, T, D/H) -> (B, T, H, D/H) -> (B, T, D)
        B, H, T, Dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)

    def forward(self, x: Tensor, causal: bool = False, mask=None) -> Tensor:
        B, T, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if causal and mask is None:
            # Lower-triangular mask shared across batch & head dims (broadcasts).
            mask = np.tril(np.ones((T, T), dtype=np.bool_))[None, None, :, :]
        out = scaled_dot_product_attention(
            q, k, v, mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = self._merge_heads(out)
        out = self.out_proj(out)
        if self.training and self.dropout_p > 0:
            out = self.resid_dropout(out)
        return out
