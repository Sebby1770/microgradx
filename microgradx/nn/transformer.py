"""Transformer building blocks.

``TransformerEncoderLayer`` mirrors the PyTorch post-norm encoder block:
Multi-head self-attention + residual + LayerNorm, then a position-wise FFN
(Linear → GELU → Dropout → Linear) + residual + LayerNorm.
"""
from __future__ import annotations

from microgradx.tensor import Tensor
from microgradx.nn.module import Module
from microgradx.nn.linear import Linear
from microgradx.nn.norm import LayerNorm
from microgradx.nn.dropout import Dropout
from microgradx.nn.activation import GELU
from microgradx.nn.attention import MultiHeadAttention


class TransformerEncoderLayer(Module):
    """A single Transformer encoder block.

    Parameters
    ----------
    d_model : int
        Input / output feature dimension.
    nhead : int
        Number of attention heads (must divide ``d_model``).
    dim_feedforward : int
        Hidden size of the position-wise FFN (default 2048).
    dropout : float
        Dropout probability applied after attention and inside the FFN.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.dim_feedforward = int(dim_feedforward)

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = GELU()

    def forward(self, src: Tensor, src_mask=None, causal: bool = False) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        src : Tensor
            Input of shape ``(N, S, d_model)``.
        src_mask : optional
            Attention mask forwarded to :class:`MultiHeadAttention`.
        causal : bool
            If True, apply a causal (lower-triangular) attention mask.
        """
        # Self-attention block (post-norm)
        attn_out = self.self_attn(src, causal=causal, mask=src_mask)
        src = self.norm1(src + self.dropout1(attn_out))
        # Feed-forward block
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src

    def __repr__(self):
        return (
            f"TransformerEncoderLayer(d_model={self.d_model}, "
            f"nhead={self.nhead}, dim_feedforward={self.dim_feedforward})"
        )
