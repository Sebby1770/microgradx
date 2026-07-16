"""GroupNorm — normalisation stats, shapes, grads."""
import numpy as np
import pytest
import microgradx as mg
from microgradx import nn


def test_groupnorm_shape():
    gn = nn.GroupNorm(num_groups=4, num_channels=8)
    x = mg.Tensor(np.random.randn(2, 8, 6, 6).astype(np.float32), requires_grad=True)
    y = gn(x)
    assert y.shape == (2, 8, 6, 6)


def test_groupnorm_normalizes_groups():
    np.random.seed(0)
    gn = nn.GroupNorm(num_groups=2, num_channels=4, affine=False)
    x = mg.Tensor((np.random.randn(3, 4, 5, 5) * 3 + 1).astype(np.float32))
    y = gn(x).numpy()
    # Within each group of 2 channels, mean≈0 var≈1 over (C_g, H, W)
    for n in range(3):
        for g in range(2):
            ch = y[n, g * 2:(g + 1) * 2, :, :]
            assert abs(ch.mean()) < 1e-4
            assert abs(ch.var() - 1.0) < 1e-2


def test_groupnorm_batch_size_one():
    gn = nn.GroupNorm(4, 8)
    x = mg.Tensor(np.random.randn(1, 8, 4, 4).astype(np.float32))
    y = gn(x)
    assert y.shape == (1, 8, 4, 4)
    assert np.isfinite(y.numpy()).all()


def test_groupnorm_rejects_bad_groups():
    with pytest.raises(ValueError):
        nn.GroupNorm(num_groups=3, num_channels=8)


def test_groupnorm_affine_grads():
    gn = nn.GroupNorm(2, 4)
    x = mg.Tensor(np.random.randn(2, 4, 3, 3).astype(np.float32))
    (gn(x) ** 2).sum().backward()
    assert gn.weight.grad is not None and gn.weight.grad.shape == (4,)
    assert gn.bias.grad is not None and gn.bias.grad.shape == (4,)
    assert np.isfinite(gn.weight.grad).all()


def test_groupnorm_input_grad():
    gn = nn.GroupNorm(2, 4)
    x = mg.Tensor(np.random.randn(2, 4, 3, 3).astype(np.float32), requires_grad=True)
    gn(x).sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
