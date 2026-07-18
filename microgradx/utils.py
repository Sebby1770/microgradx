"""Utility helpers: activation checkpointing, parameter counting, summaries, seeding."""
from __future__ import annotations

from copy import deepcopy
from functools import partial
import random
from types import ModuleType
from typing import Optional, Tuple

import numpy as np

from microgradx.autograd.function import Function
from microgradx.autograd.grad_mode import enable_grad, no_grad
from microgradx.backend import xp


def _capture_rng_state():
    """Capture the active backend RNG state when its API exposes one."""
    get_state = getattr(xp.random, "get_state", None)
    if callable(get_state):
        return "module", deepcopy(get_state())

    # CuPy exposes a device-local RandomState on supported versions. Keep this
    # capability-based so importing MicroGradX never requires CuPy.
    get_random_state = getattr(xp.random, "get_random_state", None)
    if callable(get_random_state):
        random_state = get_random_state()
        get_state = getattr(random_state, "get_state", None)
        if callable(get_state):
            return "random_state", deepcopy(get_state())
    return None


def _restore_rng_state(snapshot) -> None:
    if snapshot is None:
        return
    kind, state = snapshot
    if kind == "module":
        xp.random.set_state(state)
    else:
        xp.random.get_random_state().set_state(state)


def _discover_checkpoint_dependencies(run_fn):
    """Find Modules and trainable Tensors captured by a checkpoint callable.

    ``Function.apply`` normally decides whether to build a graph solely from
    its explicit Tensor arguments. A checkpointed Module is different: model
    inputs commonly do *not* require gradients while parameters captured by
    ``run_fn`` do. Discovering those parameters lets us attach the checkpoint
    node to the real training graph without retaining its activations.
    """
    from microgradx.nn.module import Module
    from microgradx.tensor import Tensor

    seen = set()
    module_ids = set()
    tensor_ids = set()
    modules = []
    tensors = []

    def add_tensor(tensor):
        if tensor.requires_grad and id(tensor) not in tensor_ids:
            tensor_ids.add(id(tensor))
            tensors.append(tensor)

    def add_module(module):
        for child in module.modules():
            if id(child) in module_ids:
                continue
            module_ids.add(id(child))
            modules.append(child)
            for parameter in child._parameters.values():
                add_tensor(parameter)

    def visit(obj):
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, Tensor):
            add_tensor(obj)
            return
        if isinstance(obj, Module):
            add_module(obj)
            return
        if isinstance(obj, partial):
            visit(obj.func)
            visit(obj.args)
            visit(obj.keywords or {})
            return
        if isinstance(obj, dict):
            for value in obj.values():
                visit(value)
            return
        if isinstance(obj, (tuple, list, set, frozenset)):
            for value in obj:
                visit(value)
            return
        if isinstance(obj, (
            type,
            ModuleType,
            str,
            bytes,
            int,
            float,
            complex,
            bool,
            type(None),
            xp.ndarray,
        )):
            return

        if callable(obj):
            bound_self = getattr(obj, "__self__", None)
            if bound_self is not None:
                visit(bound_self)

            for value in getattr(obj, "__defaults__", ()) or ():
                visit(value)
            visit(getattr(obj, "__kwdefaults__", None) or {})

            closure = getattr(obj, "__closure__", None) or ()
            for cell in closure:
                try:
                    visit(cell.cell_contents)
                except ValueError:  # empty closure cell
                    pass

            # A function can reference a Module stored at module scope rather
            # than in a closure. Restrict lookup to names used by its bytecode
            # so we do not walk the entire globals dictionary.
            code = getattr(obj, "__code__", None)
            globals_dict = getattr(obj, "__globals__", None)
            if code is not None and globals_dict is not None:
                for name in code.co_names:
                    if name in globals_dict:
                        visit(globals_dict[name])

        # Closures and bound methods often capture a lightweight holder rather
        # than a Module directly. Inspect explicit instance fields so those
        # indirect dependencies do not silently lose parameter gradients.
        try:
            values = vars(obj).values()
        except TypeError:
            values = ()
        for value in values:
            visit(value)

        # Slotted callable objects have no ``__dict__``. Walk only declared
        # slots (not arbitrary descriptors/properties) to discover state they
        # intentionally retain.
        for cls in type(obj).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if name in {"__dict__", "__weakref__"}:
                    continue
                try:
                    value = object.__getattribute__(obj, name)
                except (AttributeError, TypeError):
                    continue
                visit(value)

    visit(run_fn)
    return tuple(modules), tuple(tensors)


def _snapshot_buffers(modules):
    snapshots = []
    for module in modules:
        state = []
        for name, buffer in module._buffers.items():
            state.append((name, None if buffer is None else buffer.copy()))
        snapshots.append((module, tuple(state)))
    return tuple(snapshots)


def _restore_buffers(snapshots) -> None:
    for module, state in snapshots:
        current_buffers = module._buffers
        saved_names = {name for name, _ in state}

        # A recompute may register temporary state. Remove it so the module's
        # registry and public attributes exactly match the saved snapshot.
        for name in tuple(current_buffers):
            if name not in saved_names:
                current_buffers.pop(name)
                if name in module.__dict__:
                    object.__delattr__(module, name)

        restored = type(current_buffers)()
        for name, value in state:
            if value is None:
                current = None
            else:
                current = current_buffers.get(name)
                if (current is None
                        or type(current) is not type(value)
                        or current.shape != value.shape
                        or current.dtype != value.dtype):
                    current = value.copy()
                else:
                    current[...] = value
            restored[name] = current
            # Keep the public attribute synchronized even if run_fn reassigned
            # or removed it.
            object.__setattr__(module, name, current)
        module.__dict__["_buffers"] = restored


def manual_seed(seed: int) -> None:
    """Seed Python's ``random`` and NumPy RNGs for reproducibility.

        mg.manual_seed(42)
        a = mg.randn(3, 3)
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        from microgradx.backend import xp, is_gpu
        if is_gpu():
            xp.random.seed(seed)
    except Exception:
        pass


class _Checkpoint(Function):
    """Run a sub-computation without retaining its intermediate activations."""

    @staticmethod
    def forward(ctx, run_fn, run_kwargs, preserve_rng_state, modules,
                num_inputs, *flat_inputs):
        from microgradx.tensor import Tensor

        ctx.run_fn = run_fn
        ctx.run_kwargs = dict(run_kwargs)
        ctx.preserve_rng_state = bool(preserve_rng_state)
        ctx.modules = modules
        ctx.module_modes = tuple(module.training for module in modules)
        ctx.num_inputs = num_inputs
        ctx.num_captured = len(flat_inputs) - num_inputs
        # Five non-Tensor positional arguments precede the real inputs in
        # Function.apply. Preserve each input's original autograd requirement.
        ctx.input_requires_grad = ctx.needs_input_grad[5:5 + num_inputs]
        ctx.save_for_backward(*flat_inputs[:num_inputs])
        ctx.forward_buffer_state = _snapshot_buffers(modules)
        ctx.forward_rng_state = (
            _capture_rng_state() if ctx.preserve_rng_state else None
        )

        with no_grad():
            out = run_fn(
                *[Tensor(array) for array in flat_inputs[:num_inputs]],
                **ctx.run_kwargs,
            )
        if not isinstance(out, Tensor):
            raise TypeError(
                "checkpoint(fn, ...) requires fn to return a single Tensor"
            )
        return out.data

    @staticmethod
    def backward(ctx, grad_out):
        from microgradx.tensor import Tensor

        rng_at_backward = (
            _capture_rng_state() if ctx.preserve_rng_state else None
        )
        buffer_snapshots = _snapshot_buffers(ctx.modules)
        current_modes = tuple(module.training for module in ctx.modules)

        inputs = []
        try:
            # Recompute under the exact train/eval modes used by the original
            # forward, even if the caller changed mode before backward.
            for module, mode in zip(ctx.modules, ctx.module_modes):
                object.__setattr__(module, "training", mode)
            if ctx.preserve_rng_state:
                _restore_rng_state(ctx.forward_rng_state)
            _restore_buffers(ctx.forward_buffer_state)

            with enable_grad():
                inputs = [
                    Tensor(array, requires_grad=requires_grad)
                    for array, requires_grad in zip(
                        ctx.saved_tensors, ctx.input_requires_grad
                    )
                ]
                out = ctx.run_fn(*inputs, **ctx.run_kwargs)
                if not isinstance(out, Tensor):
                    raise TypeError(
                        "checkpointed fn returned a non-Tensor during recompute"
                    )
                if out.requires_grad:
                    out.backward(grad_out)
        finally:
            # A recomputation must be observationally invisible apart from
            # gradients: do not advance RNG or update registered state twice.
            _restore_buffers(buffer_snapshots)
            for module, mode in zip(ctx.modules, current_modes):
                object.__setattr__(module, "training", mode)
            if ctx.preserve_rng_state:
                _restore_rng_state(rng_at_backward)

        input_grads = tuple(
            tensor.grad if requires_grad else None
            for tensor, requires_grad in zip(inputs, ctx.input_requires_grad)
        )
        # Captured parameters receive gradients directly from the recomputed
        # graph. Returning them here as well would accumulate each one twice.
        return ((None,) * 5 + input_grads
                + (None,) * ctx.num_captured)


def checkpoint(run_fn, *args, preserve_rng_state: bool = True, **kwargs):
    """Checkpoint ``run_fn(*args, **kwargs)``.

    The forward result is identical to a direct call, but intermediate
    activations are discarded and recomputed during backward. Random-number
    state and registered Module buffers are restored around recomputation, so
    Dropout uses the original mask and BatchNorm statistics update only once.

    Tensor inputs must be positional. Trainable parameters captured by a
    Module, closure, partial, or callable object are discovered automatically,
    so ordinary data inputs do not need ``requires_grad=True``. Non-Tensor
    options may be passed as keyword arguments.
    """
    from microgradx.tensor import Tensor

    if not callable(run_fn):
        raise TypeError("checkpoint() expects a callable as its first argument")
    for index, arg in enumerate(args):
        if not isinstance(arg, Tensor):
            raise TypeError(
                "checkpoint Tensor inputs must be positional; "
                f"argument {index} is {type(arg).__name__}"
            )
    tensor_kwargs = [name for name, value in kwargs.items()
                     if isinstance(value, Tensor)]
    if tensor_kwargs:
        raise TypeError(
            "checkpoint Tensor inputs must be positional, got Tensor keyword "
            f"arguments {tensor_kwargs}"
        )

    # Stateful call options (for example ``module=layer``) participate in the
    # same dependency discovery as closures and bound callables.
    modules, captured = _discover_checkpoint_dependencies((run_fn, kwargs))
    explicit_ids = {id(arg) for arg in args}
    captured = tuple(tensor for tensor in captured
                     if id(tensor) not in explicit_ids)
    non_leaf_captured = [
        tensor for tensor in captured if tensor._ctx is not None
    ]
    if non_leaf_captured:
        raise ValueError(
            "checkpoint() captured a non-leaf Tensor; pass that Tensor as a "
            "positional input so autograd can preserve every graph branch"
        )
    return _Checkpoint.apply(
        run_fn,
        kwargs,
        preserve_rng_state,
        modules,
        len(args),
        *args,
        *captured,
    )


def count_parameters(model) -> int:
    """Total number of scalar elements across ``model.parameters()``."""
    total = 0
    for p in model.parameters():
        total += int(p.data.size)
    return total


def summary(model, input_shape: Optional[Tuple[int, ...]] = None) -> str:
    """Human-readable layer listing with per-parameter shapes and counts.

    Only leaf parameters (via ``named_parameters``) are listed; the total at
    the bottom matches :func:`count_parameters`.

        print(summary(model, input_shape=(1, 784)))
    """
    lines = [model.__class__.__name__]
    if input_shape is not None:
        lines.append(f"  input_shape: {tuple(input_shape)}")
    total = 0
    for name, p in model.named_parameters():
        n = int(p.data.size)
        total += n
        lines.append(f"  {name}: {tuple(p.shape)}  ({n:,} params)")
    # Also list registered buffers (e.g. BatchNorm running stats, Int8 weights)
    bufs = list(model.named_buffers())
    if bufs:
        lines.append("  --- buffers ---")
        for name, b in bufs:
            try:
                shape = tuple(b.shape)
                n = int(b.size)
            except Exception:
                shape = ()
                n = 0
            lines.append(f"  {name}: {shape}  ({n:,} elems)")
    lines.append(f"Total parameters: {total:,}")
    return "\n".join(lines)
