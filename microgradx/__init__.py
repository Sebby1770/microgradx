"""MicroGradX — minimalist autograd & neural-net framework on NumPy."""

from microgradx.tensor import (
    Tensor,
    zeros,
    ones,
    randn,
    rand,
    arange,
    from_numpy,
)
from microgradx.autograd import Function, Context, gradcheck, numerical_grad
from microgradx import nn, optim, data, training, export

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "zeros",
    "ones",
    "randn",
    "rand",
    "arange",
    "from_numpy",
    "Function",
    "Context",
    "gradcheck",
    "numerical_grad",
    "nn",
    "optim",
    "data",
    "training",
    "export",
]
