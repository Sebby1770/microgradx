import numpy as np
import pytest

import microgradx as mg
from microgradx import nn


def test_register_buffer_validates_names_values_and_collisions():
    module = nn.Module()
    for bad_name in ("", "nested.name"):
        with pytest.raises(KeyError):
            module.register_buffer(bad_name, np.zeros(1, dtype=np.float32))
    with pytest.raises(TypeError):
        module.register_buffer(123, np.zeros(1, dtype=np.float32))
    with pytest.raises(TypeError, match="NumPy/backend array"):
        module.register_buffer("items", [1, 2, 3])
    with pytest.raises(TypeError, match="object dtype"):
        module.register_buffer("objects", np.array([object()], dtype=object))
    with pytest.raises(TypeError, match="not Tensor"):
        module.register_buffer("tensor", mg.Tensor([1.0]))

    module.weight = mg.Tensor([1.0], requires_grad=True)
    with pytest.raises(KeyError, match="already exists"):
        module.register_buffer("weight", np.zeros(1, dtype=np.float32))
    assert dict(module.named_parameters())["weight"] is module.weight
    assert "weight" not in dict(module.named_buffers())

    module.child = nn.ReLU()
    with pytest.raises(KeyError, match="already exists"):
        module.register_buffer("child", np.zeros(1, dtype=np.float32))
    assert dict(module.named_modules())["child"] is module.child

    with pytest.raises(KeyError, match="already exists"):
        module.register_buffer("parameters", np.zeros(1, dtype=np.float32))
    assert callable(module.parameters)


def test_buffer_reassignment_keeps_registries_exclusive():
    module = nn.Module()
    module.register_buffer("state", np.zeros(2, dtype=np.float32))
    replacement = np.ones(2, dtype=np.float32)
    module.state = replacement
    assert dict(module.named_buffers())["state"] is replacement
    assert "state" not in dict(module.named_parameters())

    parameter = mg.Tensor([2.0, 3.0], requires_grad=True)
    module.state = parameter
    assert "state" not in dict(module.named_buffers())
    assert dict(module.named_parameters())["state"] is parameter
    assert module.state_dict()["state"].dtype != object

    child = nn.ReLU()
    module.state = child
    assert "state" not in dict(module.named_parameters())
    assert dict(module.named_modules())["state"] is child

    ordinary = np.full(2, 4.0, dtype=np.float32)
    module.state = ordinary
    assert "state" not in dict(module.named_modules())
    assert "state" not in dict(module.named_buffers())
    assert module.state is ordinary


def test_invalid_buffer_reassignment_is_atomic():
    module = nn.Module()
    original = np.zeros(2, dtype=np.float32)
    module.register_buffer("state", original)
    with pytest.raises(TypeError):
        module.state = [1, 2]
    assert module.state is original
    assert dict(module.named_buffers())["state"] is original


def test_buffers_walk_nested_modules_and_skip_none():
    parent = nn.Module()
    parent.register_buffer("top", np.array(1, dtype=np.int64))
    parent.register_buffer("optional", None)
    parent.child = nn.Module()
    parent.child.register_buffer("inner", np.ones(3, dtype=np.float32))

    assert list(dict(parent.named_buffers())) == ["top", "child.inner"]
    buffers = list(parent.buffers())
    assert buffers[0] is parent.top
    assert buffers[1] is parent.child.inner
