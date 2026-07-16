"""Conv1d — shapes, padding/stride, and gradient checks."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import microgradx as mg
from microgradx import nn, gradcheck

np.random.seed(1)


def test_conv1d_shape_same_padding():
    # L_out = L when padding = (k-1)//2 and stride=1 for odd k
    conv = nn.Conv1d(3, 8, kernel_size=3, stride=1, padding=1)
    x = mg.Tensor(np.random.randn(2, 3, 16).astype(np.float32), requires_grad=True)
    y = conv(x)
    assert y.shape == (2, 8, 16)


def test_conv1d_stride_padding():
    conv = nn.Conv1d(4, 6, kernel_size=5, stride=2, padding=2)
    x = mg.Tensor(np.random.randn(1, 4, 20).astype(np.float32), requires_grad=True)
    y = conv(x)
    # L_out = (20 + 4 - 5) // 2 + 1 = 10
    assert y.shape == (1, 6, 10)


def test_conv1d_no_bias():
    conv = nn.Conv1d(2, 3, kernel_size=3, bias=False)
    x = mg.Tensor(np.random.randn(2, 2, 8).astype(np.float32))
    y = conv(x)
    assert y.shape == (2, 3, 6)
    assert conv.bias is None


def test_conv1d_gradcheck_input():
    conv = nn.Conv1d(2, 3, kernel_size=3, padding=1)
    x = mg.Tensor(np.random.randn(1, 2, 7).astype(np.float64), requires_grad=True)
    # Use float64 weights for tighter numerical check.
    conv.weight.data = conv.weight.data.astype(np.float64)
    if conv.bias is not None:
        conv.bias.data = conv.bias.data.astype(np.float64)
    assert gradcheck(lambda t: conv(t).sum(), [x], eps=1e-4, atol=1e-3, rtol=1e-2)


def test_conv1d_grad_flow_params():
    conv = nn.Conv1d(2, 4, kernel_size=3, padding=1)
    x = mg.Tensor(np.random.randn(2, 2, 10).astype(np.float32), requires_grad=True)
    y = conv(x)
    y.sum().backward()
    assert x.grad is not None
    assert conv.weight.grad is not None
    assert conv.bias.grad is not None
    assert conv.weight.grad.shape == conv.weight.shape
