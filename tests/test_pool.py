"""AvgPool2d and AdaptiveAvgPool2d — shapes and simple values."""
import numpy as np
import microgradx as mg
from microgradx import nn


def test_avgpool2d_shape():
    pool = nn.AvgPool2d(2)
    x = mg.Tensor(np.random.randn(2, 3, 8, 8).astype(np.float32), requires_grad=True)
    y = pool(x)
    assert y.shape == (2, 3, 4, 4)


def test_avgpool2d_value():
    pool = nn.AvgPool2d(2)
    x = mg.Tensor(
        np.array([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]], dtype=np.float32),
        requires_grad=True,
    )
    y = pool(x)
    # 2x2 non-overlapping averages
    expected = np.array([[[[3.5, 5.5], [11.5, 13.5]]]], dtype=np.float32)
    np.testing.assert_allclose(y.numpy(), expected, atol=1e-5)


def test_avgpool2d_backward():
    pool = nn.AvgPool2d(2)
    x = mg.Tensor(np.ones((1, 1, 4, 4), dtype=np.float32), requires_grad=True)
    y = pool(x)
    y.sum().backward()
    # Each of 4 output cells contributes 1/4 to each of its 4 inputs → grad 0.25
    np.testing.assert_allclose(x.grad, np.full((1, 1, 4, 4), 0.25), atol=1e-5)


def test_avgpool2d_stride_padding_shape():
    pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    x = mg.Tensor(np.random.randn(1, 2, 5, 5).astype(np.float32))
    y = pool(x)
    assert y.shape == (1, 2, 5, 5)


def test_adaptive_avgpool2d_global():
    pool = nn.AdaptiveAvgPool2d(1)
    x = mg.Tensor(np.random.randn(4, 8, 16, 16).astype(np.float32), requires_grad=True)
    y = pool(x)
    assert y.shape == (4, 8, 1, 1)
    expected = x.numpy().mean(axis=(2, 3), keepdims=True)
    np.testing.assert_allclose(y.numpy(), expected, atol=1e-5)


def test_adaptive_avgpool2d_hw():
    pool = nn.AdaptiveAvgPool2d((2, 3))
    x = mg.Tensor(np.random.randn(2, 3, 8, 9).astype(np.float32), requires_grad=True)
    y = pool(x)
    assert y.shape == (2, 3, 2, 3)


def test_adaptive_avgpool2d_backward():
    pool = nn.AdaptiveAvgPool2d(1)
    x = mg.Tensor(np.ones((1, 1, 4, 4), dtype=np.float32), requires_grad=True)
    pool(x).sum().backward()
    # Global mean: each input gets 1/16 of upstream grad 1
    np.testing.assert_allclose(x.grad, np.full((1, 1, 4, 4), 1.0 / 16), atol=1e-5)
