import numpy as np

import microgradx as mg


def test_no_grad_disables_graph():
    x = mg.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    with mg.no_grad():
        y = (x * x).sum()
    assert not y.requires_grad
    assert y._ctx is None
    # Values are unaffected — only the graph is skipped.
    assert np.isclose(float(y.numpy()), 14.0)


def test_grad_flows_outside_no_grad():
    x = mg.Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * x).sum()
    y.backward()
    assert np.allclose(x.grad, 2 * np.array([1.0, 2.0, 3.0]))


def test_enable_grad_inside_no_grad():
    x = mg.Tensor([1.0, 2.0], requires_grad=True)
    with mg.no_grad():
        assert not mg.is_grad_enabled()
        with mg.enable_grad():
            assert mg.is_grad_enabled()
            z = (x * x).sum()
        assert z.requires_grad
    # Flag is correctly restored after the with-blocks.
    assert mg.is_grad_enabled()


def test_no_grad_as_decorator():
    @mg.no_grad()
    def predict(t):
        return t * 2

    x = mg.Tensor([1.0], requires_grad=True)
    out = predict(x)
    assert not out.requires_grad
    # Decorator must restore the global flag even if it was on.
    assert mg.is_grad_enabled()


def test_set_grad_enabled_returns_previous():
    prev = mg.set_grad_enabled(False)
    assert prev is True
    assert not mg.is_grad_enabled()
    mg.set_grad_enabled(prev)
    assert mg.is_grad_enabled()
