"""TransformerEncoder stack — multi-layer forward shape and grads."""
import numpy as np
import microgradx as mg
from microgradx import nn


def test_transformer_encoder_shape():
    layer = nn.TransformerEncoderLayer(
        d_model=32, nhead=4, dim_feedforward=64, dropout=0.0
    )
    enc = nn.TransformerEncoder(layer, num_layers=3, norm=nn.LayerNorm(32))
    enc.eval()
    x = mg.Tensor(np.random.randn(2, 7, 32).astype(np.float32), requires_grad=True)
    y = enc(x)
    assert y.shape == (2, 7, 32)


def test_transformer_encoder_from_config():
    enc = nn.TransformerEncoder.from_config(
        d_model=16, nhead=2, num_layers=2, dim_feedforward=32, dropout=0.0
    )
    enc.eval()
    x = mg.Tensor(np.random.randn(3, 5, 16).astype(np.float32))
    y = enc(x)
    assert y.shape == (3, 5, 16)
    assert enc.num_layers == 2
    assert enc.norm is not None


def test_transformer_encoder_from_config_no_norm():
    enc = nn.TransformerEncoder.from_config(
        16, 2, num_layers=1, final_norm=False, dropout=0.0
    )
    assert enc.norm is None
    y = enc(mg.Tensor(np.random.randn(1, 4, 16).astype(np.float32)))
    assert y.shape == (1, 4, 16)


def test_transformer_encoder_grad():
    enc = nn.TransformerEncoder.from_config(
        d_model=16, nhead=2, num_layers=2, dim_feedforward=32, dropout=0.0
    )
    enc.train()
    x = mg.Tensor(np.random.randn(2, 4, 16).astype(np.float32), requires_grad=True)
    y = enc(x)
    y.sum().backward()
    assert x.grad is not None
    assert np.all(np.isfinite(x.grad))
    assert any(p.grad is not None for p in enc.parameters())


def test_transformer_encoder_num_layers():
    layer = nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0.0)
    enc = nn.TransformerEncoder(layer, num_layers=4)
    assert len(enc.layers) == 4
    assert enc.num_layers == 4
