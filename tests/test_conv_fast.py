"""Validate the stride-trick Conv2d forward against a naive reference, and
confirm the gradient still checks out after the im2col rewrite."""
import numpy as np

import microgradx as mg
from microgradx import nn
from microgradx.autograd import gradcheck


def naive_conv2d(x, w, b, stride, padding):
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = w.shape
    sh, sw = stride
    ph, pw = padding
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    OH = (H + 2 * ph - KH) // sh + 1
    OW = (W + 2 * pw - KW) // sw + 1
    out = np.zeros((N, Cout, OH, OW), dtype=np.float64)
    for n in range(N):
        for co in range(Cout):
            for oh in range(OH):
                for ow in range(OW):
                    region = xp[n, :, oh * sh:oh * sh + KH, ow * sw:ow * sw + KW]
                    out[n, co, oh, ow] = np.sum(region * w[co]) + (b[co] if b is not None else 0)
    return out


def test_conv_forward_matches_naive_strided_padded():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 7, 7)).astype(np.float32)
    conv = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)

    w = conv.weight.numpy()
    b = conv.bias.numpy() if conv.bias is not None else None
    ref = naive_conv2d(x.astype(np.float64), w.astype(np.float64),
                       None if b is None else b.astype(np.float64),
                       conv.stride, conv.padding)

    got = conv(mg.Tensor(x)).numpy()
    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-4)


def test_conv_gradcheck_after_rewrite():
    conv = nn.Conv2d(2, 3, kernel_size=2, stride=1, padding=0)

    def fn(x):
        return conv(x).sum()

    x = mg.Tensor(np.random.default_rng(1).standard_normal((1, 2, 5, 5)),
                  requires_grad=True)
    assert gradcheck(fn, (x,))
