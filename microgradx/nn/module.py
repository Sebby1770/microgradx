"""
Module — base class for stateful, composable layers.

Mirrors torch.nn.Module conventions:
  - subclass and override `forward`
  - assign `Tensor`s with requires_grad=True OR child `Module`s as attrs;
    `__setattr__` auto-tracks them
  - `parameters()` walks the whole subtree
  - `train()/eval()` flips a flag that ops like Dropout read
"""
from __future__ import annotations
from typing import Iterator, Tuple, Dict, Any, List
from collections import OrderedDict

import numpy as np

from microgradx.tensor import Tensor


class Module:
    def __init__(self):
        # Use __dict__ so __setattr__ doesn't recurse on these.
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "training", True)

    # ---- attribute hooks ----
    def __setattr__(self, name, value):
        # Lazy-init in case subclass forgot super().__init__()
        if "_parameters" not in self.__dict__:
            object.__setattr__(self, "_parameters", OrderedDict())
            object.__setattr__(self, "_modules", OrderedDict())
            object.__setattr__(self, "_buffers", OrderedDict())
            object.__setattr__(self, "training", True)
        params = self.__dict__["_parameters"]
        modules = self.__dict__["_modules"]
        # Drop stale registrations if reassigning
        params.pop(name, None)
        modules.pop(name, None)
        if isinstance(value, Tensor) and value.requires_grad:
            params[name] = value
        elif isinstance(value, Module):
            modules[name] = value
        object.__setattr__(self, name, value)

    # ---- traversal ----
    def parameters(self) -> Iterator[Tensor]:
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Tensor]]:
        for n, p in self._parameters.items():
            yield (f"{prefix}{n}" if prefix else n), p
        for n, m in self._modules.items():
            sub_prefix = f"{prefix}{n}." if prefix else f"{n}."
            yield from m.named_parameters(sub_prefix)

    def modules(self) -> Iterator["Module"]:
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        yield prefix, self
        for n, m in self._modules.items():
            sub_prefix = f"{prefix}.{n}" if prefix else n
            yield from m.named_modules(sub_prefix)

    # ---- mode ----
    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    # ---- gradient management ----
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    # ---- callable ----
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    # ---- state dict (for save/load) ----
    def state_dict(self) -> Dict[str, Any]:
        sd = OrderedDict()
        for n, p in self.named_parameters():
            sd[n] = p.numpy()
        return sd

    def load_state_dict(self, sd: Dict[str, Any], strict: bool = True):
        """Copy parameters from `sd` (name → array) into this module.

        With `strict=True` (default) the key sets must match exactly and every
        tensor's shape must agree. Returns self for chaining.
        """
        own = dict(self.named_parameters())
        if strict:
            missing = sorted(set(own) - set(sd))
            unexpected = sorted(set(sd) - set(own))
            if missing or unexpected:
                raise KeyError(
                    f"state_dict mismatch: missing={missing}, "
                    f"unexpected={unexpected}"
                )
        for k, v in sd.items():
            p = own.get(k)
            if p is None:
                continue  # non-strict: ignore extras
            arr = np.asarray(v)
            if arr.shape != tuple(p.data.shape):
                raise ValueError(
                    f"shape mismatch for {k!r}: got {arr.shape}, "
                    f"expected {tuple(p.data.shape)}"
                )
            p.data = arr.astype(p.data.dtype, copy=True)
        return self

    def save(self, path):
        """Write this module's weights to `path` (a `.npz` file)."""
        from microgradx.serialization import save as _save
        return _save(self, path)

    def load(self, path, strict: bool = True):
        """Load weights from `path` into this module in place. Returns self."""
        from microgradx.serialization import load as _load
        return self.load_state_dict(_load(path), strict=strict)

    def __repr__(self):
        lines = [self.__class__.__name__ + "("]
        for n, m in self._modules.items():
            for sub in repr(m).split("\n"):
                lines.append(f"  ({n}): {sub}" if sub == repr(m).split("\n")[0]
                             else f"      {sub}")
        lines.append(")")
        return "\n".join(lines)


class Sequential(Module):
    """Chains modules in order. Accepts positional Module args or a list."""

    def __init__(self, *modules):
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            modules = tuple(modules[0])
        for i, m in enumerate(modules):
            setattr(self, str(i), m)
        self._n = len(modules)

    def forward(self, x):
        for i in range(self._n):
            x = getattr(self, str(i))(x)
        return x


class ModuleList(Module):
    """Like Python list, but registers children for parameter discovery."""

    def __init__(self, modules=()):
        super().__init__()
        self._n = 0
        for m in modules:
            self.append(m)

    def append(self, m: Module):
        setattr(self, str(self._n), m)
        self._n += 1

    def __getitem__(self, i: int) -> Module:
        return getattr(self, str(i))

    def __iter__(self):
        for i in range(self._n):
            yield getattr(self, str(i))

    def __len__(self):
        return self._n
