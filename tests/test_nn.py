"""Layers — shape, gradient correctness, train/eval mode."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import microgradx as mg
from microgradx import nn, gradcheck

np.random.seed(0)


def test_linear_shape_and_grad():
    lin = nn.Linear(8, 4)
    x = mg.Tensor(np.random.randn(3, 8), requires_grad=True)
    y = lin(x)
    assert y.shape == (3, 4)
    # Input gradient
    assert gradcheck(lambda x: lin(x).sum(), [x])
    # Weight gradient — must use a *fixed* x captured in the closure
    fixed_x = mg.Tensor(np.random.randn(2, 8))
    assert gradcheck(lambda w: (fixed_x @ w.transpose()).sum(), [lin.weight])


def test_layernorm():
    ln = nn.LayerNorm(16)
    x = mg.Tensor(np.random.randn(2, 5, 16), requires_grad=True)
    y = ln(x)
    # Output should have ≈ zero mean and ≈ unit var across feature dim
    np.testing.assert_allclose(y.numpy().mean(axis=-1), 0, atol=1e-5)
    np.testing.assert_allclose(y.numpy().var(axis=-1), 1, atol=1e-3)
    assert gradcheck(lambda x: ln(x).sum(), [x])


def test_conv2d_shape_padding_stride():
    conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
    x = mg.Tensor(np.random.randn(2, 3, 5, 5), requires_grad=True)
    y = conv(x)
    assert y.shape == (2, 8, 5, 5)
    assert gradcheck(lambda x: conv(x).sum(), [x])
    # Weight grad — keep input fixed
    fixed = mg.Tensor(np.random.randn(2, 3, 5, 5))
    assert gradcheck(lambda w: conv(fixed).sum(), [conv.weight])


def test_conv2d_stride2():
    conv = nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1)
    x = mg.Tensor(np.random.randn(1, 2, 8, 8), requires_grad=True)
    y = conv(x)
    assert y.shape == (1, 4, 4, 4)


def test_maxpool():
    pool = nn.MaxPool2d(2)
    x = mg.Tensor(np.array([[[[1, 2, 3, 4],
                               [5, 6, 7, 8],
                               [9, 10, 11, 12],
                               [13, 14, 15, 16]]]], dtype=np.float32),
                  requires_grad=True)
    y = pool(x)
    np.testing.assert_array_equal(y.numpy(), [[[[6, 8], [14, 16]]]])
    y.sum().backward()
    # Gradient should land on max positions only
    assert int(x.grad.sum()) == 4  # 4 max selections, each gets grad 1


def test_dropout_train_eval():
    drop = nn.Dropout(0.5)
    x = mg.Tensor(np.ones((1000,)), requires_grad=True)
    drop.train()
    y_train = drop(x).numpy()
    # Should be either 0 or 2.0 (1/(1-0.5))
    assert set(np.unique(y_train.round(4))) <= {0.0, 2.0}
    drop.eval()
    y_eval = drop(x).numpy()
    np.testing.assert_array_equal(y_eval, x.numpy())


def test_cross_entropy_value():
    # Hand-computed: uniform logits → -log(1/C)
    C = 4
    logits = mg.Tensor(np.zeros((10, C)))
    target = np.zeros(10, dtype=np.int64)
    loss = nn.cross_entropy(logits, target)
    expected = -np.log(1.0 / C)
    np.testing.assert_allclose(float(loss.data), expected, atol=1e-5)


def test_attention_causal_mask():
    """Token at position t must not be able to attend to t+k (k>0)."""
    mha = nn.MultiHeadAttention(d_model=8, n_heads=2)
    mha.eval()
    B, T = 1, 4
    x = mg.Tensor(np.random.randn(B, T, 8) * 0.1, requires_grad=True)
    y = mha(x, causal=True)

    # Perturb a future token; loss for an earlier token should be unchanged
    y0_before = mha(x, causal=True).numpy()[:, 0, :].copy()
    x.data[:, 2, :] += 1.0
    y0_after = mha(x, causal=True).numpy()[:, 0, :]
    np.testing.assert_allclose(y0_before, y0_after, atol=1e-5)


def test_param_discovery():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    params = list(model.parameters())
    # Each Linear has weight + bias → 4 params
    assert len(params) == 4
