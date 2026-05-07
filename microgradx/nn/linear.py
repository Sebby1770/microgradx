"""Linear (fully connected) layer.

Math:
  y = x @ Wᵀ + b,  where W has shape (out_features, in_features)
  x can be any leading-batch shape; only the last dim is contracted.
"""
import math
import numpy as np

from microgradx.tensor import Tensor
from microgradx.nn.module import Module
from microgradx.nn import init


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(
            np.zeros((out_features, in_features), dtype=np.float32),
            requires_grad=True,
        )
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            bound = 1.0 / math.sqrt(in_features)
            self.bias = Tensor(
                np.random.uniform(-bound, bound, size=(out_features,)).astype(np.float32),
                requires_grad=True,
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return (f"Linear(in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None})")
