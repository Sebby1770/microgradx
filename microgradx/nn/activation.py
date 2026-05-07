"""Activation functions as nn Modules (most just delegate to ops)."""
from microgradx.tensor import Tensor
from microgradx.autograd.function import Function
from microgradx.nn.module import Module
from microgradx.autograd.ops import GELU as _GELUFn


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return _GELUFn.apply(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(axis=self.axis)
