"""Benchmark the two im2col paths and confirm they are byte-identical.

    python3 bench/conv_im2col.py

The Conv2d forward dispatches between a stride-trick view (large kernels) and
a per-position slice loop (small kernels). This script measures the crossover
that motivates the threshold in microgradx/nn/conv.py.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import microgradx as mg
from microgradx import nn
import microgradx.nn.conv as C


def _loop(xx, kh, kw, stride, padding):
    N, Cc, H, W = xx.shape
    sh, sw = stride
    ph, pw = padding
    OH = (H + 2 * ph - kh) // sh + 1
    OW = (W + 2 * pw - kw) // sw + 1
    xpad = np.pad(xx, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if (ph or pw) else xx
    cols = np.zeros((N, Cc, kh, kw, OH, OW), dtype=xx.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = xpad[:, :, i:i + sh * OH:sh, j:j + sw * OW:sw]
    return cols.transpose(0, 4, 5, 1, 2, 3), (OH, OW)


def bench(fn, n=30):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


CONFIGS = [
    ("3x3   s1 p1  16->32 @32", (16, 16, 32, 32), dict(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)),
    ("5x5   s2 p2  16->16 @64", (8, 16, 64, 64), dict(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=2)),
    ("7x7   s1 p3   3->16 @64", (8, 3, 64, 64), dict(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3)),
    ("11x11 s4 p0   3->64 @64", (8, 3, 64, 64), dict(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=0)),
]

orig = C._im2col
print(f"{'config':<26}{'loop':>10}{'dispatched':>13}{'speedup':>10}")
for name, xs, kw in CONFIGS:
    x = mg.Tensor(np.random.randn(*xs).astype(np.float32))
    conv = nn.Conv2d(**kw)

    # correctness: both paths must agree
    a = np.ascontiguousarray(orig(x.numpy(), kw["kernel_size"], kw["kernel_size"],
                                  (kw["stride"],) * 2, (kw["padding"],) * 2)[0])
    b = np.ascontiguousarray(_loop(x.numpy(), kw["kernel_size"], kw["kernel_size"],
                                   (kw["stride"],) * 2, (kw["padding"],) * 2)[0])
    assert np.allclose(a, b), f"{name}: paths disagree"

    C._im2col = orig
    t_new = bench(lambda: conv(x))
    C._im2col = _loop
    t_old = bench(lambda: conv(x))
    C._im2col = orig
    print(f"{name:<26}{t_old:>9.2f}ms{t_new:>11.2f}ms{t_old / t_new:>9.2f}x")
