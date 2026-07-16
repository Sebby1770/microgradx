"""RNN / GRU / LSTM — shapes, batch_first, multi-layer, grad flow."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import microgradx as mg
from microgradx import nn

np.random.seed(0)


def test_rnn_shapes_seq_first():
    T, B, F, H = 7, 3, 5, 8
    x = mg.Tensor(np.random.randn(T, B, F).astype(np.float32), requires_grad=True)
    m = nn.RNN(F, H, num_layers=1)
    out, h_n = m(x)
    assert out.shape == (T, B, H)
    assert h_n.shape == (1, B, H)


def test_rnn_batch_first_and_multilayer():
    B, T, F, H = 2, 6, 4, 10
    x = mg.Tensor(np.random.randn(B, T, F).astype(np.float32), requires_grad=True)
    m = nn.RNN(F, H, num_layers=2, batch_first=True)
    out, h_n = m(x)
    assert out.shape == (B, T, H)
    assert h_n.shape == (2, B, H)


def test_gru_shapes_and_grad():
    T, B, F, H = 5, 2, 6, 9
    x = mg.Tensor(np.random.randn(T, B, F).astype(np.float32), requires_grad=True)
    m = nn.GRU(F, H, num_layers=2)
    out, h_n = m(x)
    assert out.shape == (T, B, H)
    assert h_n.shape == (2, B, H)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    # Parameters should have gradients.
    for p in m.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.shape


def test_lstm_shapes_and_grad():
    B, T, F, H = 3, 4, 7, 11
    x = mg.Tensor(np.random.randn(B, T, F).astype(np.float32), requires_grad=True)
    m = nn.LSTM(F, H, num_layers=1, batch_first=True)
    out, (h_n, c_n) = m(x)
    assert out.shape == (B, T, H)
    assert h_n.shape == (1, B, H)
    assert c_n.shape == (1, B, H)
    loss = out.mean() + h_n.sum() * 0.01
    loss.backward()
    assert x.grad is not None
    for p in m.parameters():
        assert p.grad is not None


def test_rnn_with_initial_hidden():
    T, B, F, H = 3, 2, 4, 5
    x = mg.Tensor(np.random.randn(T, B, F).astype(np.float32))
    h0 = mg.Tensor(np.random.randn(1, B, H).astype(np.float32))
    m = nn.RNN(F, H)
    out, h_n = m(x, h0)
    assert out.shape == (T, B, H)
    assert h_n.shape == (1, B, H)


def test_lstm_with_initial_state():
    T, B, F, H = 3, 2, 4, 5
    x = mg.Tensor(np.random.randn(T, B, F).astype(np.float32), requires_grad=True)
    h0 = mg.Tensor(np.random.randn(1, B, H).astype(np.float32), requires_grad=True)
    c0 = mg.Tensor(np.random.randn(1, B, H).astype(np.float32), requires_grad=True)
    m = nn.LSTM(F, H)
    out, (h_n, c_n) = m(x, (h0, c0))
    out.sum().backward()
    assert h0.grad is not None
    assert c0.grad is not None


def test_gru_relu_rnn_no_crash():
    m = nn.RNN(4, 6, nonlinearity="relu")
    x = mg.Tensor(np.random.randn(2, 1, 4).astype(np.float32), requires_grad=True)
    out, _ = m(x)
    out.sum().backward()
    assert x.grad is not None
