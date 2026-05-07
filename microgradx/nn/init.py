"""Weight initialisers — Kaiming, Xavier, etc. — operating on Tensors in-place."""
import numpy as np
import math

from microgradx.tensor import Tensor


def _calc_fan(shape):
    if len(shape) < 2:
        return shape[0], shape[0]
    fan_in = shape[1] * int(np.prod(shape[2:])) if len(shape) > 2 else shape[1]
    fan_out = shape[0] * int(np.prod(shape[2:])) if len(shape) > 2 else shape[0]
    return fan_in, fan_out


def kaiming_uniform_(t: Tensor, a: float = math.sqrt(5)):
    """Kaiming-He uniform — default init for Linear layers."""
    fan_in, _ = _calc_fan(t.shape)
    gain = math.sqrt(2.0 / (1 + a * a))
    bound = gain * math.sqrt(3.0 / fan_in)
    t.data = np.random.uniform(-bound, bound, size=t.shape).astype(t.data.dtype)
    return t


def kaiming_normal_(t: Tensor, mode: str = "fan_in"):
    fan_in, fan_out = _calc_fan(t.shape)
    fan = fan_in if mode == "fan_in" else fan_out
    std = math.sqrt(2.0 / fan)
    t.data = np.random.normal(0, std, size=t.shape).astype(t.data.dtype)
    return t


def xavier_uniform_(t: Tensor, gain: float = 1.0):
    fan_in, fan_out = _calc_fan(t.shape)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    t.data = np.random.uniform(-bound, bound, size=t.shape).astype(t.data.dtype)
    return t


def normal_(t: Tensor, mean: float = 0.0, std: float = 1.0):
    t.data = np.random.normal(mean, std, size=t.shape).astype(t.data.dtype)
    return t


def zeros_(t: Tensor):
    t.data = np.zeros(t.shape, dtype=t.data.dtype)
    return t


def ones_(t: Tensor):
    t.data = np.ones(t.shape, dtype=t.data.dtype)
    return t


def uniform_(t: Tensor, low=0.0, high=1.0):
    t.data = np.random.uniform(low, high, size=t.shape).astype(t.data.dtype)
    return t
