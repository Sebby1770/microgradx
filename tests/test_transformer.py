"""TransformerEncoderLayer — forward shape (N, S, D) → same."""
import numpy as np
import microgradx as mg
from microgradx import nn


def test_transformer_encoder_layer_shape():
    layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0)
    layer.eval()
    x = mg.Tensor(np.random.randn(2, 7, 32).astype(np.float32), requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 7, 32)


def test_transformer_encoder_layer_grad():
    layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, dim_feedforward=32, dropout=0.0)
    layer.train()
    x = mg.Tensor(np.random.randn(3, 5, 16).astype(np.float32), requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert np.all(np.isfinite(x.grad))
    # Some parameters should have grads
    has_grad = any(p.grad is not None for p in layer.parameters())
    assert has_grad


def test_transformer_encoder_layer_causal():
    layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0)
    layer.eval()
    x = mg.Tensor(np.random.randn(1, 6, 16).astype(np.float32))
    y = layer(x, causal=True)
    assert y.shape == (1, 6, 16)
