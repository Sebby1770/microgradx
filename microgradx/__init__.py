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
from microgradx.autograd import (
    Function,
    Context,
    gradcheck,
    numerical_grad,
    no_grad,
    enable_grad,
    is_grad_enabled,
    set_grad_enabled,
)
from microgradx.serialization import save, load
from microgradx.utils import checkpoint, count_parameters, summary, manual_seed
from microgradx.logging import CSVLogger
from microgradx.training.early_stopping import EarlyStopping
from microgradx.training.ema import EMA
from microgradx.metrics import accuracy
from microgradx import nn, optim, data, training, export, quant, metrics

__version__ = "0.6.0"

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
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    "set_grad_enabled",
    "save",
    "load",
    "checkpoint",
    "count_parameters",
    "summary",
    "manual_seed",
    "CSVLogger",
    "EarlyStopping",
    "EMA",
    "accuracy",
    "nn",
    "optim",
    "data",
    "training",
    "export",
    "quant",
    "metrics",
]
