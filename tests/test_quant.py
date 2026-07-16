"""Dynamic INT8 quantisation — close outputs, Linear replacement."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import microgradx as mg
from microgradx import nn
from microgradx.quant import quantize_dynamic, Int8Linear, Observer

np.random.seed(0)


def test_observer_absmax_scale():
    obs = Observer(qmax=127)
    w = np.array([[-2.0, 1.0], [0.5, 2.0]], dtype=np.float32)
    q, scale = obs.quantize(w)
    assert np.isclose(scale, 2.0 / 127.0)
    assert q.dtype == np.int8
    recon = q.astype(np.float32) * scale
    # Max relative error for symmetric absmax quant is small.
    np.testing.assert_allclose(recon, w, atol=scale)


def test_int8_linear_close_to_fp32():
    lin = nn.Linear(16, 8)
    # Deterministic weights in a reasonable range.
    lin.weight.data = np.random.randn(8, 16).astype(np.float32) * 0.1
    lin.bias.data = np.random.randn(8).astype(np.float32) * 0.01

    x_np = np.random.randn(4, 16).astype(np.float32)
    x = mg.Tensor(x_np)
    y_fp = lin(x).numpy()

    qlin = Int8Linear.from_linear(lin)
    y_q = qlin(mg.Tensor(x_np)).numpy()

    # Weight-only INT8 should stay close on small activations.
    np.testing.assert_allclose(y_q, y_fp, rtol=0.05, atol=0.05)


def test_quantize_dynamic_replaces_linears():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    x = mg.Tensor(np.random.randn(3, 10).astype(np.float32))
    y_fp = model(x).numpy()

    qmodel = quantize_dynamic(model)
    # Children should be Int8Linear where Linear was.
    assert isinstance(qmodel._modules["0"], Int8Linear)
    assert isinstance(qmodel._modules["2"], Int8Linear)
    # ReLU untouched.
    assert isinstance(qmodel._modules["1"], nn.ReLU)

    y_q = qmodel(mg.Tensor(np.random.randn(3, 10).astype(np.float32) * 0 + x.numpy())).numpy()
    np.testing.assert_allclose(y_q, y_fp, rtol=0.1, atol=0.1)


def test_quantize_dynamic_top_level_linear():
    lin = nn.Linear(4, 2)
    q = quantize_dynamic(lin)
    assert isinstance(q, Int8Linear)
    x = mg.Tensor(np.random.randn(1, 4).astype(np.float32))
    assert q(x).shape == (1, 2)
