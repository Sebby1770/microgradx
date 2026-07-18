import numpy as np
import pytest

import microgradx as mg
from microgradx import nn


def test_checkpoint_matches_direct():
    """Checkpointing must be numerically transparent: identical forward output,
    identical input gradients, and identical parameter gradients."""
    np.random.seed(0)
    block = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))

    # --- direct ---
    x1 = mg.Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
    out1 = block(x1)
    (out1 * out1).sum().backward()
    g_x1 = x1.grad.copy()
    g_w1 = [p.grad.copy() for p in block.parameters()]

    # --- checkpointed (same weights, fresh grads) ---
    block.zero_grad()
    x2 = mg.Tensor(x1.numpy(), requires_grad=True)
    out2 = mg.checkpoint(lambda t: block(t), x2)
    (out2 * out2).sum().backward()

    assert np.allclose(out1.numpy(), out2.numpy())
    assert np.allclose(g_x1, x2.grad, atol=1e-5)
    for a, p in zip(g_w1, block.parameters()):
        assert np.allclose(a, p.grad, atol=1e-5)


def test_checkpoint_forward_is_grad_free():
    # The output of a checkpointed region still participates in autograd.
    x = mg.Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    y = mg.checkpoint(lambda t: t * 3.0, x)
    assert y.requires_grad
    y.sum().backward()
    assert np.allclose(x.grad, 3.0)


def test_checkpoint_rejects_multiple_outputs():
    x = mg.Tensor(np.zeros((2, 2), dtype=np.float32), requires_grad=True)
    try:
        mg.checkpoint(lambda t: (t, t), x)
    except TypeError:
        return
    raise AssertionError("expected TypeError for multi-output run_fn")


def test_checkpoint_module_parameters_work_with_ordinary_input():
    """Training data does not normally require grad; module parameters do."""
    np.random.seed(3)
    layer = nn.Linear(4, 3)
    x_data = np.random.randn(5, 4).astype(np.float32)
    x = mg.Tensor(x_data)  # intentionally requires_grad=False

    out = mg.checkpoint(layer, x)
    assert out.requires_grad
    out.sum().backward()

    expected_w = np.broadcast_to(x_data.sum(axis=0), layer.weight.shape)
    np.testing.assert_allclose(layer.weight.grad, expected_w, atol=1e-6)
    np.testing.assert_allclose(layer.bias.grad, np.full((3,), 5.0), atol=1e-6)
    assert x.grad is None


def test_checkpoint_discovers_closure_parameters_and_forwards_kwargs():
    np.random.seed(4)
    layer = nn.Linear(3, 2, bias=False)
    x = mg.Tensor(np.random.randn(4, 3).astype(np.float32))

    def scaled_layer(t, scale=1.0):
        return layer(t) * scale

    mg.checkpoint(scaled_layer, x, scale=2.5).sum().backward()
    expected = np.broadcast_to(2.5 * x.data.sum(axis=0), layer.weight.shape)
    np.testing.assert_allclose(layer.weight.grad, expected, atol=1e-6)


def test_checkpoint_discovers_module_dependencies_passed_by_keyword():
    layer = nn.Linear(3, 2, bias=False)
    x_data = np.arange(12, dtype=np.float32).reshape(4, 3)

    def apply_layer(value, module=None):
        return module(value)

    out = mg.checkpoint(apply_layer, mg.Tensor(x_data), module=layer)
    assert out.requires_grad
    out.sum().backward()

    expected = np.broadcast_to(x_data.sum(axis=0), layer.weight.shape)
    np.testing.assert_allclose(layer.weight.grad, expected, atol=1e-6)


def test_checkpoint_discovers_parameters_on_slotted_callable_objects():
    class SlottedLayer:
        __slots__ = ("layer",)

        def __init__(self):
            self.layer = nn.Linear(3, 2, bias=False)

        def __call__(self, value):
            return self.layer(value)

    fn = SlottedLayer()
    x_data = np.arange(12, dtype=np.float32).reshape(4, 3)
    out = mg.checkpoint(fn, mg.Tensor(x_data))
    assert out.requires_grad
    out.sum().backward()

    expected = np.broadcast_to(x_data.sum(axis=0), fn.layer.weight.shape)
    np.testing.assert_allclose(fn.layer.weight.grad, expected, atol=1e-6)


def test_checkpoint_discovers_modules_through_holder_objects():
    class Holder:
        def __init__(self):
            self.layer = nn.Linear(3, 2, bias=False)

        def run(self, value):
            return self.layer(value)

    x_data = np.arange(12, dtype=np.float32).reshape(4, 3)
    expected = np.broadcast_to(x_data.sum(axis=0), (2, 3))

    bound_holder = Holder()
    bound_out = mg.checkpoint(bound_holder.run, mg.Tensor(x_data))
    assert bound_out.requires_grad
    bound_out.sum().backward()
    np.testing.assert_allclose(
        bound_holder.layer.weight.grad, expected, atol=1e-6
    )

    closure_holder = Holder()

    def through_closure(value):
        return closure_holder.layer(value)

    closure_out = mg.checkpoint(through_closure, mg.Tensor(x_data))
    assert closure_out.requires_grad
    closure_out.sum().backward()
    np.testing.assert_allclose(
        closure_holder.layer.weight.grad, expected, atol=1e-6
    )


def test_checkpoint_replays_dropout_rng_without_advancing_global_stream():
    drop = nn.Dropout(0.35)
    data = np.linspace(-2, 2, 40, dtype=np.float32)
    upstream = mg.Tensor(np.linspace(0.25, 1.25, 40, dtype=np.float32))

    np.random.seed(123)
    direct_x = mg.Tensor(data.copy(), requires_grad=True)
    direct_out = drop(direct_x)
    (direct_out * upstream).sum().backward()

    np.random.seed(123)
    checkpoint_x = mg.Tensor(data.copy(), requires_grad=True)
    checkpoint_out = mg.checkpoint(drop, checkpoint_x)
    # Predict what the next draws should be after the original forward. The
    # recomputation in backward must not consume from this ambient stream.
    shadow_rng = np.random.RandomState()
    shadow_rng.set_state(np.random.get_state())
    expected_next = shadow_rng.rand(8)
    (checkpoint_out * upstream).sum().backward()
    actual_next = np.random.rand(8)

    np.testing.assert_array_equal(checkpoint_out.numpy(), direct_out.numpy())
    np.testing.assert_array_equal(checkpoint_x.grad, direct_x.grad)
    np.testing.assert_array_equal(actual_next, expected_next)


def test_checkpoint_restores_batchnorm_buffers_and_forward_mode():
    np.random.seed(5)
    direct = nn.BatchNorm1d(3, momentum=0.5)
    checked = nn.BatchNorm1d(3, momentum=0.5)
    x_data = (np.random.randn(8, 3) * 2 + 4).astype(np.float32)

    direct_out = direct(mg.Tensor(x_data))
    (direct_out * direct_out).sum().backward()

    checked_out = mg.checkpoint(checked, mg.Tensor(x_data))
    stats_after_forward = (
        checked.running_mean.copy(),
        checked.running_var.copy(),
        checked.num_batches_tracked.copy(),
    )
    # Recompute with the original train mode, but restore the caller's current
    # eval mode after backward.
    checked.eval()
    (checked_out * checked_out).sum().backward()

    assert not checked.training
    np.testing.assert_array_equal(checked.running_mean, stats_after_forward[0])
    np.testing.assert_array_equal(checked.running_var, stats_after_forward[1])
    np.testing.assert_array_equal(
        checked.num_batches_tracked, stats_after_forward[2]
    )
    np.testing.assert_allclose(checked.weight.grad, direct.weight.grad, atol=1e-5)
    np.testing.assert_allclose(checked.bias.grad, direct.bias.grad, atol=1e-5)


def test_checkpoint_recomputes_from_forward_buffer_state():
    class BufferedScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", np.array(2.0, dtype=np.float32))

        def forward(self, value):
            return value * self.scale.item()

    layer = BufferedScale()
    x = mg.Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
    out = mg.checkpoint(layer, x)

    # Caller changes after the forward must survive backward, while the
    # recomputation itself must still use the scale that produced `out`.
    layer.scale[...] = 5.0
    out.sum().backward()

    np.testing.assert_array_equal(x.grad, np.full(3, 2.0, dtype=np.float32))
    assert layer.scale.item() == 5.0


def test_checkpoint_respects_no_grad_with_captured_parameters():
    layer = nn.Linear(3, 2)
    with mg.no_grad():
        out = mg.checkpoint(layer, mg.Tensor(np.ones((2, 3), dtype=np.float32)))
    assert not out.requires_grad
    assert out._ctx is None


def test_checkpoint_rejects_untracked_tensor_arguments():
    x = mg.Tensor(np.ones((2, 2), dtype=np.float32), requires_grad=True)
    with pytest.raises(TypeError, match="must be positional"):
        mg.checkpoint(lambda t: t * 2, np.ones((2, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="Tensor keyword"):
        mg.checkpoint(lambda t, other=None: t + other, x, other=x)


def test_checkpoint_requires_captured_non_leaf_as_positional_input():
    x = mg.Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    intermediate = x * 2

    with pytest.raises(ValueError, match="non-leaf Tensor"):
        mg.checkpoint(lambda: intermediate * 3)

    checked = mg.checkpoint(lambda value: value * 3, intermediate)
    (checked.sum() + intermediate.sum()).backward()
    np.testing.assert_array_equal(
        x.grad,
        np.array([8.0, 8.0], dtype=np.float32),
    )
