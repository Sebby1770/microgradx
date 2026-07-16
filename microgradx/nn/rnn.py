"""
Recurrent layers — RNN, GRU, LSTM.

PyTorch-ish API. Input layout is (seq_len, batch, input_size) by default, or
(batch, seq_len, input_size) with ``batch_first=True``.

Math (single layer, single step):

  RNN:  h' = tanh(x W_ihᵀ + b_ih + h W_hhᵀ + b_hh)

  GRU:  r = σ(x W_irᵀ + …),  z = σ(x W_izᵀ + …)
        n = tanh(x W_inᵀ + r ⊙ (h W_hnᵀ + …))
        h' = (1 − z) ⊙ n + z ⊙ h

  LSTM: i,f,g,o gates; c' = f ⊙ c + i ⊙ g; h' = o ⊙ tanh(c')

Weights for multi-gate cells are packed along the first dim
(GRU: r/z/n, LSTM: i/f/g/o), matching PyTorch layout.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np

from microgradx.tensor import Tensor
from microgradx.autograd.ops import Concat
from microgradx.nn.module import Module
from microgradx.nn.dropout import Dropout


def _uniform_k(t: Tensor, hidden_size: int) -> None:
    """PyTorch default RNN init: U(-1/√H, 1/√H)."""
    k = 1.0 / math.sqrt(hidden_size)
    t.data = np.random.uniform(-k, k, size=t.shape).astype(np.float32)


def _stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Stack rank-R tensors into rank-(R+1) along ``axis`` via Concat."""
    if not tensors:
        raise ValueError("cannot stack empty list")
    if len(tensors) == 1:
        t = tensors[0]
        shape = list(t.shape)
        shape.insert(axis if axis >= 0 else axis + t.ndim + 1, 1)
        return t.reshape(*shape)
    expanded = []
    for t in tensors:
        shape = list(t.shape)
        shape.insert(axis if axis >= 0 else axis + t.ndim + 1, 1)
        expanded.append(t.reshape(*shape))
    return Concat.apply(*expanded, axis=axis)


class _RNNBase(Module):
    """Shared multi-layer recurrent machinery."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        gate_size: int = 1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.gate_size = gate_size
        self._drop = Dropout(dropout) if dropout > 0 and num_layers > 1 else None

        for layer in range(num_layers):
            layer_in = input_size if layer == 0 else hidden_size
            w_ih = Tensor(
                np.zeros((gate_size * hidden_size, layer_in), dtype=np.float32),
                requires_grad=True,
            )
            w_hh = Tensor(
                np.zeros((gate_size * hidden_size, hidden_size), dtype=np.float32),
                requires_grad=True,
            )
            _uniform_k(w_ih, hidden_size)
            _uniform_k(w_hh, hidden_size)
            setattr(self, f"weight_ih_l{layer}", w_ih)
            setattr(self, f"weight_hh_l{layer}", w_hh)
            if bias:
                b_ih = Tensor(
                    np.zeros(gate_size * hidden_size, dtype=np.float32),
                    requires_grad=True,
                )
                b_hh = Tensor(
                    np.zeros(gate_size * hidden_size, dtype=np.float32),
                    requires_grad=True,
                )
                _uniform_k(b_ih, hidden_size)
                _uniform_k(b_hh, hidden_size)
                setattr(self, f"bias_ih_l{layer}", b_ih)
                setattr(self, f"bias_hh_l{layer}", b_hh)
            else:
                setattr(self, f"bias_ih_l{layer}", None)
                setattr(self, f"bias_hh_l{layer}", None)

    def _get_layer_params(self, layer: int):
        return (
            getattr(self, f"weight_ih_l{layer}"),
            getattr(self, f"weight_hh_l{layer}"),
            getattr(self, f"bias_ih_l{layer}"),
            getattr(self, f"bias_hh_l{layer}"),
        )

    def _linear(self, x: Tensor, w: Tensor, b: Optional[Tensor]) -> Tensor:
        out = x @ w.transpose()
        if b is not None:
            out = out + b
        return out

    def _prepare_input(self, x: Tensor) -> Tuple[Tensor, int, int, int, bool]:
        if x.ndim != 3:
            raise ValueError(
                f"RNN input must be 3-D, got shape {x.shape}"
            )
        if self.batch_first:
            # (B, T, F) → (T, B, F)
            x = x.transpose(1, 0, 2)
        T, B, F = x.shape
        if F != self.input_size:
            raise ValueError(
                f"expected input_size={self.input_size}, got last dim {F}"
            )
        return x, T, B, F, self.batch_first

    def _zeros_hidden(self, batch: int) -> Tensor:
        return Tensor(
            np.zeros((self.num_layers, batch, self.hidden_size), dtype=np.float32)
        )

    def _cell(
        self,
        x_t: Tensor,
        h_prev: Tensor,
        w_ih: Tensor,
        w_hh: Tensor,
        b_ih: Optional[Tensor],
        b_hh: Optional[Tensor],
        c_prev: Optional[Tensor] = None,
    ):
        raise NotImplementedError

    def _run_sequence(
        self,
        x: Tensor,
        hx: Optional[Tensor],
        cx: Optional[Tensor] = None,
        return_cell: bool = False,
    ):
        x, T, B, _, batch_first = self._prepare_input(x)

        if hx is None:
            hx = self._zeros_hidden(B)
        if return_cell and cx is None:
            cx = self._zeros_hidden(B)

        # Per-layer hidden (and cell) at current time.
        h_layers: List[Tensor] = [hx[i] for i in range(self.num_layers)]
        c_layers: List[Optional[Tensor]] = (
            [cx[i] for i in range(self.num_layers)] if return_cell else [None] * self.num_layers
        )

        # Collect outputs of the top layer for each timestep.
        top_outs: List[Tensor] = []

        for t in range(T):
            x_t = x[t]  # (B, F) or (B, H)
            for layer in range(self.num_layers):
                w_ih, w_hh, b_ih, b_hh = self._get_layer_params(layer)
                if return_cell:
                    h_new, c_new = self._cell(
                        x_t, h_layers[layer], w_ih, w_hh, b_ih, b_hh, c_layers[layer]
                    )
                    h_layers[layer] = h_new
                    c_layers[layer] = c_new
                else:
                    h_new = self._cell(
                        x_t, h_layers[layer], w_ih, w_hh, b_ih, b_hh
                    )
                    h_layers[layer] = h_new
                x_t = h_new
                # Dropout between layers (not after last), training only.
                if (
                    self._drop is not None
                    and layer < self.num_layers - 1
                    and self.training
                ):
                    x_t = self._drop(x_t)
            top_outs.append(x_t)

        # (T, B, H)
        output = _stack(top_outs, axis=0)
        h_n = _stack(h_layers, axis=0)  # (num_layers, B, H)

        if batch_first:
            output = output.transpose(1, 0, 2)

        if return_cell:
            c_n = _stack(c_layers, axis=0)  # type: ignore[arg-type]
            return output, h_n, c_n
        return output, h_n


class RNN(_RNNBase):
    """Elman RNN with tanh (or relu) nonlinearity.

    Returns ``(output, h_n)`` where:
      - output: (seq_len, batch, hidden) or (batch, seq_len, hidden)
      - h_n:    (num_layers, batch, hidden)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ):
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity must be 'tanh' or 'relu'")
        super().__init__(
            input_size, hidden_size, num_layers, bias, batch_first, dropout, gate_size=1
        )
        self.nonlinearity = nonlinearity

    def _cell(self, x_t, h_prev, w_ih, w_hh, b_ih, b_hh, c_prev=None):
        pre = self._linear(x_t, w_ih, b_ih) + self._linear(h_prev, w_hh, b_hh)
        if self.nonlinearity == "relu":
            return pre.relu()
        return pre.tanh()

    def forward(
        self, x: Tensor, hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self._run_sequence(x, hx)  # type: ignore[return-value]

    def __repr__(self):
        return (
            f"RNN({self.input_size}, {self.hidden_size}, "
            f"num_layers={self.num_layers}, nonlinearity={self.nonlinearity!r}, "
            f"bias={self.bias}, batch_first={self.batch_first}, "
            f"dropout={self.dropout})"
        )


class GRU(_RNNBase):
    """Gated Recurrent Unit.

    Returns ``(output, h_n)`` with the same shapes as :class:`RNN`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__(
            input_size, hidden_size, num_layers, bias, batch_first, dropout, gate_size=3
        )

    def _cell(self, x_t, h_prev, w_ih, w_hh, b_ih, b_hh, c_prev=None):
        H = self.hidden_size
        gi = self._linear(x_t, w_ih, b_ih)   # (B, 3H)
        gh = self._linear(h_prev, w_hh, b_hh)

        # Split packed gates: reset, update, new
        i_r, i_z, i_n = gi[:, :H], gi[:, H:2 * H], gi[:, 2 * H:]
        h_r, h_z, h_n = gh[:, :H], gh[:, H:2 * H], gh[:, 2 * H:]

        r = (i_r + h_r).sigmoid()
        z = (i_z + h_z).sigmoid()
        n = (i_n + r * h_n).tanh()
        # h' = (1 - z) * n + z * h
        return (Tensor(1.0) - z) * n + z * h_prev

    def forward(
        self, x: Tensor, hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self._run_sequence(x, hx)  # type: ignore[return-value]

    def __repr__(self):
        return (
            f"GRU({self.input_size}, {self.hidden_size}, "
            f"num_layers={self.num_layers}, bias={self.bias}, "
            f"batch_first={self.batch_first}, dropout={self.dropout})"
        )


class LSTM(_RNNBase):
    """Long Short-Term Memory.

    Returns ``(output, (h_n, c_n))`` where each of ``h_n`` / ``c_n`` has shape
    ``(num_layers, batch, hidden_size)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__(
            input_size, hidden_size, num_layers, bias, batch_first, dropout, gate_size=4
        )

    def _cell(self, x_t, h_prev, w_ih, w_hh, b_ih, b_hh, c_prev=None):
        H = self.hidden_size
        gates = self._linear(x_t, w_ih, b_ih) + self._linear(h_prev, w_hh, b_hh)
        # i, f, g, o
        i = gates[:, :H].sigmoid()
        f = gates[:, H:2 * H].sigmoid()
        g = gates[:, 2 * H:3 * H].tanh()
        o = gates[:, 3 * H:].sigmoid()
        c_new = f * c_prev + i * g
        h_new = o * c_new.tanh()
        return h_new, c_new

    def forward(
        self,
        x: Tensor,
        hx: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if hx is None:
            h0, c0 = None, None
        elif isinstance(hx, (tuple, list)):
            h0, c0 = hx
        else:
            raise TypeError("LSTM hx must be None or a (h_0, c_0) tuple")
        output, h_n, c_n = self._run_sequence(x, h0, c0, return_cell=True)
        return output, (h_n, c_n)

    def __repr__(self):
        return (
            f"LSTM({self.input_size}, {self.hidden_size}, "
            f"num_layers={self.num_layers}, bias={self.bias}, "
            f"batch_first={self.batch_first}, dropout={self.dropout})"
        )
