"""Nearest-neighbor Upsample / interpolate."""
import numpy as np
import microgradx as mg
from microgradx import nn


def test_upsample_nearest_2x_doubles_hw():
    up = nn.Upsample(scale_factor=2, mode="nearest")
    x = mg.Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32), requires_grad=True)
    y = up(x)
    assert y.shape == (2, 3, 8, 10)


def test_upsample_nearest_values():
    up = nn.Upsample(scale_factor=2, mode="nearest")
    x = mg.Tensor(
        np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        requires_grad=True,
    )
    y = up(x)
    expected = np.array(
        [[[[1, 1, 2, 2],
           [1, 1, 2, 2],
           [3, 3, 4, 4],
           [3, 3, 4, 4]]]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(y.numpy(), expected)


def test_upsample_backward():
    up = nn.Upsample(scale_factor=2, mode="nearest")
    x = mg.Tensor(np.ones((1, 1, 2, 2), dtype=np.float32), requires_grad=True)
    y = up(x)
    y.sum().backward()
    # Each input cell is repeated 2x2 = 4 times → grad 4
    np.testing.assert_allclose(x.grad, np.full((1, 1, 2, 2), 4.0), atol=1e-5)


def test_interpolate_scale_factor():
    x = mg.Tensor(np.random.randn(1, 2, 3, 3).astype(np.float32))
    y = nn.interpolate(x, scale_factor=3, mode="nearest")
    assert y.shape == (1, 2, 9, 9)


def test_upsample_asymmetric_scale():
    up = nn.Upsample(scale_factor=(2, 3), mode="nearest")
    x = mg.Tensor(np.random.randn(1, 1, 4, 4).astype(np.float32))
    y = up(x)
    assert y.shape == (1, 1, 8, 12)
