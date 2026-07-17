"""Transformer building blocks.

``TransformerEncoderLayer`` mirrors the PyTorch post-norm encoder block:
Multi-head self-attention + residual + LayerNorm, then a position-wise FFN
(Linear → GELU → Dropout → Linear) + residual + LayerNorm.

``TransformerEncoder`` stacks N layers with optional final LayerNorm.
``PositionalEncoding`` adds fixed sinusoidal PE (Vaswani et al.) + dropout.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from microgradx.tensor import Tensor
from microgradx.nn.module import Module, ModuleList
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
        self.dropout_p = float(dropout)

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


class TransformerEncoder(Module):
    """Stack of :class:`TransformerEncoderLayer` with optional final LayerNorm.

    Construct either from a prototype layer::

        layer = TransformerEncoderLayer(d_model=64, nhead=4)
        enc = TransformerEncoder(layer, num_layers=4)

    or from configuration kwargs via :meth:`from_config`::

        enc = TransformerEncoder.from_config(64, 4, num_layers=4)

    Parameters
    ----------
    encoder_layer : TransformerEncoderLayer
        Prototype layer; ``num_layers`` independent copies are built with the
        same hyperparameters (fresh weights each).
    num_layers : int
        Number of encoder layers.
    norm : Module or None
        Optional final normalisation (typically ``LayerNorm(d_model)``).
    """

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Optional[Module] = None,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.num_layers = int(num_layers)
        self.layers = ModuleList(
            [
                TransformerEncoderLayer(
                    encoder_layer.d_model,
                    encoder_layer.nhead,
                    dim_feedforward=encoder_layer.dim_feedforward,
                    dropout=encoder_layer.dropout_p,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm = norm

    @classmethod
    def from_config(
        cls,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        final_norm: bool = True,
    ) -> "TransformerEncoder":
        """Build an encoder stack from hyperparameters."""
        layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        norm = LayerNorm(d_model) if final_norm else None
        return cls(layer, num_layers, norm=norm)

    def forward(self, src: Tensor, src_mask=None, causal: bool = False) -> Tensor:
        """Run ``src`` through every layer; apply final norm if set.

        Input / output shape: ``(N, S, d_model)``.
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, causal=causal)
        if self.norm is not None:
            output = self.norm(output)
        return output

    def __repr__(self):
        return (
            f"TransformerEncoder(num_layers={self.num_layers}, "
            f"norm={self.norm is not None})"
        )


class PositionalEncoding(Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017) + dropout.

    PE is fixed (non-learnable buffer). Supports ``batch_first`` layouts:

    - ``batch_first=True``  (default): input ``(B, T, D)``, PE along dim 1
    - ``batch_first=False``: input ``(T, B, D)``, PE along dim 0

    Parameters
    ----------
    d_model : int
        Embedding / model dimension.
    max_len : int
        Maximum sequence length precomputed.
    dropout : float
        Dropout applied after adding PE.
    batch_first : bool
        Layout convention (see above).
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_len < 1:
            raise ValueError(f"max_len must be positive, got {max_len}")
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.batch_first = bool(batch_first)
        self.dropout = Dropout(dropout)

        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len, dtype=np.float32)[:, None]  # (T, 1)
        div_term = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)

        if batch_first:
            # (1, T, D) — broadcast over batch
            pe = pe[None, :, :]
        else:
            # (T, 1, D) — broadcast over batch
            pe = pe[:, None, :]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add PE to ``x`` then apply dropout.

        Raises
        ------
        ValueError
            If sequence length exceeds ``max_len``.
        """
        if self.batch_first:
            # x: (B, T, D)
            t = x.shape[1]
            if t > self.max_len:
                raise ValueError(
                    f"sequence length {t} exceeds max_len={self.max_len}"
                )
            pe_slice = self.pe[:, :t, :]
        else:
            # x: (T, B, D)
            t = x.shape[0]
            if t > self.max_len:
                raise ValueError(
                    f"sequence length {t} exceeds max_len={self.max_len}"
                )
            pe_slice = self.pe[:t, :, :]
        return self.dropout(x + pe_slice)

    def __repr__(self):
        return (
            f"PositionalEncoding(d_model={self.d_model}, "
            f"max_len={self.max_len}, batch_first={self.batch_first})"
        )
