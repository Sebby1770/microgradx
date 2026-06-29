import numpy as np
import pytest

import microgradx as mg
from microgradx import nn
from microgradx.autograd import gradcheck


def test_batchnorm2d_normalizes_in_train():
    np.random.seed(1)
    bn = nn.BatchNorm2d(3)
    bn.train()
    x = mg.Tensor((np.random.randn(8, 3, 4, 4) * 5 + 2).astype(np.float32))
    y = bn(x).numpy()
    for c in range(3):
        ch = y[:, c, :, :]
        assert abs(ch.mean()) < 1e-4          # centered
        assert abs(ch.var() - 1.0) < 1e-2     # unit variance


def test_batchnorm1d_accepts_2d_and_3d():
    bn = nn.BatchNorm1d(4)
    bn.train()
    assert bn(mg.Tensor(np.random.randn(6, 4).astype(np.float32))).shape == (6, 4)
    assert bn(mg.Tensor(np.random.randn(6, 4, 5).astype(np.float32))).shape == (6, 4, 5)


def test_batchnorm2d_rejects_wrong_dims():
    with pytest.raises(ValueError):
        nn.BatchNorm2d(3)(mg.Tensor(np.random.randn(8, 3).astype(np.float32)))


def test_batchnorm_running_stats_update():
    bn = nn.BatchNorm1d(3)
    bn.train()
    assert np.allclose(bn.running_mean, 0.0)
    assert np.allclose(bn.running_var, 1.0)
    bn(mg.Tensor((np.random.randn(32, 3) + 5).astype(np.float32)))
    # running mean should have moved toward the batch mean (~5)
    assert (bn.running_mean > 0.1).all()


def test_batchnorm_eval_uses_running_stats():
    np.random.seed(2)
    bn = nn.BatchNorm1d(4)
    bn.train()
    for _ in range(10):
        bn(mg.Tensor(np.random.randn(16, 4).astype(np.float32)))
    bn.eval()
    rm, rv = bn.running_mean.copy(), bn.running_var.copy()

    x = mg.Tensor(np.random.randn(16, 4).astype(np.float32))
    y = bn(x).numpy()
    expected = (x.numpy() - rm) / np.sqrt(rv + bn.eps)  # affine identity at init
    assert np.allclose(y, expected, atol=1e-5)
    # eval must not mutate running stats
    assert np.allclose(bn.running_mean, rm)
    assert np.allclose(bn.running_var, rv)


def test_batchnorm_eval_single_sample():
    # A single eval sample must work (the failure mode BatchNorm running stats
    # exist to solve).
    bn = nn.BatchNorm1d(3)
    bn.train()
    for _ in range(5):
        bn(mg.Tensor(np.random.randn(16, 3).astype(np.float32)))
    bn.eval()
    out = bn(mg.Tensor(np.random.randn(1, 3).astype(np.float32)))
    assert out.shape == (1, 3)
    assert np.isfinite(out.numpy()).all()


def test_batchnorm_gradcheck_input():
    bn = nn.BatchNorm1d(3)
    bn.train()
    x = mg.Tensor(np.random.randn(6, 3), requires_grad=True)
    assert gradcheck(lambda t: bn(t).sum(), (x,))


def test_batchnorm_affine_params_get_grads():
    bn = nn.BatchNorm1d(3)
    bn.train()
    x = mg.Tensor(np.random.randn(8, 3).astype(np.float32))
    (bn(x) * bn(x)).sum().backward()
    assert bn.weight.grad is not None and bn.weight.grad.shape == (3,)
    assert bn.bias.grad is not None and bn.bias.grad.shape == (3,)
