"""Extra activations — LeakyReLU, SiLU, Softplus shapes + simple backward."""
import numpy as np
import microgradx as mg
from microgradx import nn


def test_leaky_relu_positive_identity():
    act = nn.LeakyReLU(0.01)
    x = mg.Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), requires_grad=True)
    y = act(x)
    np.testing.assert_allclose(y.numpy(), [1.0, 2.0, 3.0], atol=1e-6)


def test_leaky_relu_negative_slope():
    act = nn.LeakyReLU(0.1)
    x = mg.Tensor(np.array([-2.0, -1.0, 0.0, 1.0], dtype=np.float32), requires_grad=True)
    y = act(x)
    np.testing.assert_allclose(y.numpy(), [-0.2, -0.1, 0.0, 1.0], atol=1e-6)


def test_leaky_relu_backward():
    act = nn.LeakyReLU(0.1)
    x = mg.Tensor(np.array([-2.0, 3.0], dtype=np.float32), requires_grad=True)
    act(x).sum().backward()
    # d/dx = slope on negative, 1 on positive
    np.testing.assert_allclose(x.grad, [0.1, 1.0], atol=1e-5)


def test_silu_shape_and_value():
    act = nn.SiLU()
    x = mg.Tensor(np.array([0.0, 1.0, -1.0], dtype=np.float32), requires_grad=True)
    y = act(x)
    assert y.shape == (3,)
    # silu(0)=0; silu(1)=1*sigmoid(1); silu(-1)=-sigmoid(-1)
    sig = 1.0 / (1.0 + np.exp(-x.numpy()))
    np.testing.assert_allclose(y.numpy(), x.numpy() * sig, atol=1e-5)


def test_silu_backward():
    act = nn.SiLU()
    x = mg.Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
    y = act(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert np.isfinite(x.grad).all()


def test_softplus_shape_and_value():
    act = nn.Softplus()
    x = mg.Tensor(np.array([0.0, 1.0, -1.0, 10.0], dtype=np.float32), requires_grad=True)
    y = act(x)
    expected = np.log1p(np.exp(np.clip(x.numpy(), None, 20)))
    # For large positive x softplus ≈ x
    expected[-1] = 10.0
    np.testing.assert_allclose(y.numpy()[:3], expected[:3], atol=1e-5)
    np.testing.assert_allclose(y.numpy()[-1], 10.0, atol=1e-3)


def test_softplus_backward():
    act = nn.Softplus()
    x = mg.Tensor(np.array([-1.0, 0.0, 2.0], dtype=np.float32), requires_grad=True)
    act(x).sum().backward()
    # grad = sigmoid(x)
    expected = 1.0 / (1.0 + np.exp(-x.numpy()))
    np.testing.assert_allclose(x.grad, expected, atol=1e-5)
