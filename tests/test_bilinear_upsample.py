"""Bilinear Upsample / interpolate."""
import numpy as np
import microgradx as mg
from microgradx import nn


def test_bilinear_shape():
    up = nn.Upsample(scale_factor=2, mode="bilinear")
    x = mg.Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32), requires_grad=True)
    y = up(x)
    assert y.shape == (2, 3, 8, 10)


def test_bilinear_identity_at_grid():
    """At even output indices (multiples of scale) values match input pixels."""
    up = nn.Upsample(scale_factor=2, mode="bilinear")
    x_np = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
    x = mg.Tensor(x_np)
    y = up(x).numpy()
    # oh=0 → src 0, ow=0 → src 0
    assert np.isclose(y[0, 0, 0, 0], 1.0)
    # oh=0, ow=2 → src w=1
    assert np.isclose(y[0, 0, 0, 2], 2.0)
    # oh=2, ow=0 → src h=1
    assert np.isclose(y[0, 0, 2, 0], 3.0)
    # oh=2, ow=2 → src (1,1)
    assert np.isclose(y[0, 0, 2, 2], 4.0)


def test_bilinear_midpoint():
    up = nn.Upsample(scale_factor=2, mode="bilinear")
    x = mg.Tensor(np.array([[[[0.0, 2.0], [0.0, 2.0]]]], dtype=np.float32))
    y = up(x).numpy()
    # Mid between 0 and 2 along width at row 0: should be ~1
    assert np.isclose(y[0, 0, 0, 1], 1.0, atol=1e-5)


def test_bilinear_backward():
    up = nn.Upsample(scale_factor=2, mode="bilinear")
    x = mg.Tensor(np.ones((1, 1, 2, 2), dtype=np.float32), requires_grad=True)
    y = up(x)
    y.sum().backward()
    assert x.grad is not None
    assert np.all(np.isfinite(x.grad))
    # Total grad mass should equal number of output elements (sum of ones out)
    np.testing.assert_allclose(x.grad.sum(), y.data.size, atol=1e-4)


def test_interpolate_bilinear():
    x = mg.Tensor(np.random.randn(1, 2, 3, 3).astype(np.float32))
    y = nn.interpolate(x, scale_factor=3, mode="bilinear")
    assert y.shape == (1, 2, 9, 9)
