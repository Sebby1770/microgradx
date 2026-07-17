"""PositionalEncoding — shape, deterministic, batch_first layouts."""
import numpy as np
import microgradx as mg
from microgradx import nn


def test_pe_shape_batch_first():
    pe = nn.PositionalEncoding(d_model=32, max_len=100, dropout=0.0, batch_first=True)
    pe.eval()
    x = mg.Tensor(np.zeros((4, 10, 32), dtype=np.float32), requires_grad=True)
    y = pe(x)
    assert y.shape == (4, 10, 32)


def test_pe_shape_time_first():
    pe = nn.PositionalEncoding(d_model=16, max_len=50, dropout=0.0, batch_first=False)
    pe.eval()
    x = mg.Tensor(np.zeros((8, 3, 16), dtype=np.float32))
    y = pe(x)
    assert y.shape == (8, 3, 16)


def test_pe_deterministic():
    pe = nn.PositionalEncoding(d_model=8, max_len=20, dropout=0.0)
    pe.eval()
    x = mg.Tensor(np.zeros((1, 5, 8), dtype=np.float32))
    y1 = pe(x).numpy().copy()
    y2 = pe(x).numpy().copy()
    np.testing.assert_allclose(y1, y2)
    # PE values should be non-zero (sin/cos of positions)
    assert np.any(np.abs(y1) > 1e-6)


def test_pe_matches_formula():
    """First even dim is sin(pos / 10000^{0/d}), first odd is cos(...)."""
    d_model = 4
    pe_mod = nn.PositionalEncoding(d_model, max_len=10, dropout=0.0)
    pe_mod.eval()
    x = mg.Tensor(np.zeros((1, 3, d_model), dtype=np.float32))
    y = pe_mod(x).numpy()[0]  # (3, 4)

    for pos in range(3):
        for i in range(0, d_model, 2):
            div = 10000.0 ** (i / d_model)
            expected_sin = np.sin(pos / div)
            expected_cos = np.cos(pos / div)
            np.testing.assert_allclose(y[pos, i], expected_sin, atol=1e-5)
            np.testing.assert_allclose(y[pos, i + 1], expected_cos, atol=1e-5)


def test_pe_adds_to_input():
    pe = nn.PositionalEncoding(d_model=8, max_len=16, dropout=0.0)
    pe.eval()
    base = np.ones((2, 4, 8), dtype=np.float32)
    x = mg.Tensor(base.copy())
    y = pe(x).numpy()
    # y should equal 1 + PE
    pe_only = pe(mg.Tensor(np.zeros_like(base))).numpy()
    np.testing.assert_allclose(y, base + pe_only, atol=1e-5)
