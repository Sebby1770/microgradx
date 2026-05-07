"""
Tensor — the central type. Holds data, an optional gradient, and a pointer
into the computation graph (`_ctx`) describing how it was produced.

Backward pass:
  1. Iterative DFS topological sort starting at the loss tensor
  2. Seed `loss.grad` (defaults to 1 for scalar losses)
  3. Walk the topo order in reverse, asking each node's saved Function for
     gradients w.r.t. its inputs, then accumulating into those inputs

Broadcasting is handled centrally in `_unbroadcast`: every op can return a
"raw" gradient and we shrink it back to the input shape by summing along
the broadcast axes. This keeps the per-op backward code dramatically simpler.
"""
from __future__ import annotations
from typing import Optional, Tuple, Union, List
import numpy as np

from microgradx.backend import xp, asarray, to_numpy

ArrayLike = Union[np.ndarray, list, tuple, float, int]


def _unbroadcast(grad, target_shape):
    """Reduce `grad` back to `target_shape` by summing along broadcasted axes.

    Two phases:
      1. Drop leading axes that exist only because of broadcasting
         (e.g. (3, 4) broadcast against (5, 3, 4) ⇒ leading dim 5 must go)
      2. For axes where target is 1 but grad is >1, sum with keepdims
    """
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """N-dimensional array with optional autograd tracking."""

    __slots__ = ("data", "requires_grad", "grad", "_ctx", "_retain_grad")

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        dtype=None,
    ):
        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, (xp.ndarray, np.generic)):
            arr = xp.asarray(data)
            self.data = arr if dtype is None else arr.astype(dtype, copy=False)
        else:
            # Python lists / scalars: default float dtypes to fp32 (ML convention).
            arr = asarray(data, dtype=dtype if dtype else None)
            if arr.dtype == np.float64 and dtype is None:
                arr = arr.astype(np.float32, copy=False)
            self.data = arr
        self.requires_grad = bool(requires_grad)
        self.grad: Optional[xp.ndarray] = None
        self._ctx = None  # Context object set by Function.apply
        self._retain_grad = False

    # -------- shape / dtype / repr --------
    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(self.data.size)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __len__(self):
        return self.data.shape[0]

    def __repr__(self):
        rg = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({to_numpy(self.data)!r}{rg})"

    def numpy(self) -> np.ndarray:
        return to_numpy(self.data)

    def item(self):
        return float(to_numpy(self.data))

    def detach(self) -> "Tensor":
        """Return a Tensor sharing data but cut off from the graph."""
        out = Tensor(self.data, requires_grad=False)
        return out

    def retain_grad(self):
        """Keep .grad alive on a non-leaf tensor (mirrors PyTorch semantics)."""
        self._retain_grad = True

    # -------- autograd entry point --------
    def backward(self, grad: Optional[ArrayLike] = None):
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require grad")

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad must be specified for non-scalar outputs "
                    f"(got shape {self.shape})"
                )
            grad = xp.ones_like(self.data)
        elif not isinstance(grad, xp.ndarray):
            grad = asarray(grad, dtype=self.data.dtype)

        # Iterative topological sort. Recursion blows the stack on graphs
        # that any real model produces.
        topo: List[Tensor] = []
        visited = set()
        stack: List[Tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, processed = stack.pop()
            node_id = id(node)
            if processed:
                topo.append(node)
                continue
            if node_id in visited:
                continue
            visited.add(node_id)
            stack.append((node, True))
            if node._ctx is not None:
                for inp in node._ctx.input_tensors:
                    if inp is not None and id(inp) not in visited:
                        stack.append((inp, False))

        self.grad = grad if self.grad is None else self.grad + grad

        for v in reversed(topo):
            if v._ctx is None or v.grad is None:
                continue
            ctx = v._ctx
            grads = ctx.fn.backward(ctx, v.grad)
            if not isinstance(grads, (tuple, list)):
                grads = (grads,)
            for inp, g in zip(ctx.input_tensors, grads):
                if inp is None or not inp.requires_grad or g is None:
                    continue
                g = _unbroadcast(g, inp.data.shape)
                inp.grad = g if inp.grad is None else inp.grad + g
            # Free graph storage for non-retained intermediates
            if not v._retain_grad and v is not self:
                v._ctx = None

    def zero_grad(self):
        self.grad = None

    # -------- operator overloads (delegated to ops module) --------
    def __add__(self, other):
        from microgradx.autograd.ops import Add
        return Add.apply(self, _ensure_tensor(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from microgradx.autograd.ops import Sub
        return Sub.apply(self, _ensure_tensor(other))

    def __rsub__(self, other):
        from microgradx.autograd.ops import Sub
        return Sub.apply(_ensure_tensor(other), self)

    def __mul__(self, other):
        from microgradx.autograd.ops import Mul
        return Mul.apply(self, _ensure_tensor(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from microgradx.autograd.ops import Div
        return Div.apply(self, _ensure_tensor(other))

    def __rtruediv__(self, other):
        from microgradx.autograd.ops import Div
        return Div.apply(_ensure_tensor(other), self)

    def __neg__(self):
        from microgradx.autograd.ops import Neg
        return Neg.apply(self)

    def __pow__(self, other):
        from microgradx.autograd.ops import Pow
        if isinstance(other, Tensor):
            raise NotImplementedError("Tensor**Tensor not implemented; use scalar exponent")
        return Pow.apply(self, exponent=float(other))

    def __matmul__(self, other):
        from microgradx.autograd.ops import MatMul
        return MatMul.apply(self, _ensure_tensor(other))

    def __getitem__(self, key):
        from microgradx.autograd.ops import GetItem
        return GetItem.apply(self, key=key)

    # -------- shape-shifting (autograd-aware) --------
    def reshape(self, *shape):
        from microgradx.autograd.ops import Reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Reshape.apply(self, shape=shape)

    def view(self, *shape):
        return self.reshape(*shape)

    def transpose(self, *axes):
        from microgradx.autograd.ops import Transpose
        if not axes:
            axes = tuple(range(self.ndim))[::-1]
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return Transpose.apply(self, axes=axes)

    def permute(self, *axes):
        return self.transpose(*axes)

    def expand(self, *shape):
        from microgradx.autograd.ops import Expand
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Expand.apply(self, shape=shape)

    # -------- reductions --------
    def sum(self, axis=None, keepdims=False):
        from microgradx.autograd.ops import Sum
        return Sum.apply(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        from microgradx.autograd.ops import Mean
        return Mean.apply(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        from microgradx.autograd.ops import Max
        return Max.apply(self, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        from microgradx.autograd.ops import Min
        return Min.apply(self, axis=axis, keepdims=keepdims)

    # -------- elementwise --------
    def exp(self):
        from microgradx.autograd.ops import Exp
        return Exp.apply(self)

    def log(self):
        from microgradx.autograd.ops import Log
        return Log.apply(self)

    def sqrt(self):
        from microgradx.autograd.ops import Sqrt
        return Sqrt.apply(self)

    def tanh(self):
        from microgradx.autograd.ops import Tanh
        return Tanh.apply(self)

    def sigmoid(self):
        from microgradx.autograd.ops import Sigmoid
        return Sigmoid.apply(self)

    def relu(self):
        from microgradx.autograd.ops import ReLU
        return ReLU.apply(self)

    def softmax(self, axis=-1):
        from microgradx.autograd.ops import Softmax
        return Softmax.apply(self, axis=axis)

    def log_softmax(self, axis=-1):
        from microgradx.autograd.ops import LogSoftmax
        return LogSoftmax.apply(self, axis=axis)

    def matmul(self, other):
        return self.__matmul__(other)


def _ensure_tensor(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


# ---- factory helpers ----
def zeros(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(xp.zeros(shape, dtype=dtype or np.float32), requires_grad=requires_grad)


def ones(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(xp.ones(shape, dtype=dtype or np.float32), requires_grad=requires_grad)


def randn(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    arr = xp.random.randn(*shape).astype(dtype or np.float32)
    return Tensor(arr, requires_grad=requires_grad)


def rand(*shape, dtype=None, requires_grad=False):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    arr = xp.random.rand(*shape).astype(dtype or np.float32)
    return Tensor(arr, requires_grad=requires_grad)


def arange(start, stop=None, step=1, dtype=None, requires_grad=False):
    if stop is None:
        start, stop = 0, start
    return Tensor(xp.arange(start, stop, step, dtype=dtype), requires_grad=requires_grad)


def from_numpy(arr, requires_grad=False):
    return Tensor(asarray(arr), requires_grad=requires_grad)
