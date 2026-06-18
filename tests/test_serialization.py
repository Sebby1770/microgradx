import numpy as np
import pytest

import microgradx as mg
from microgradx import nn


def make_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_save_load_roundtrip(tmp_path):
    model = make_model()
    path = tmp_path / "model.npz"
    mg.save(model, path)

    fresh = make_model()
    # Sanity: a fresh model has different weights.
    a = model.state_dict()
    b = fresh.state_dict()
    assert not all(np.allclose(a[k], b[k]) for k in a)

    fresh.load_state_dict(mg.load(path))
    a = model.state_dict()
    b = fresh.state_dict()
    for k in a:
        assert np.allclose(a[k], b[k]), k


def test_module_convenience_methods(tmp_path):
    model = make_model()
    path = str(tmp_path / "w.npz")
    model.save(path)
    fresh = make_model().load(path)
    for k, v in model.state_dict().items():
        assert np.allclose(fresh.state_dict()[k], v)


def test_strict_key_mismatch_raises(tmp_path):
    model = make_model()
    sd = model.state_dict()
    sd["bogus.param"] = np.zeros(3, dtype=np.float32)
    with pytest.raises(KeyError):
        make_model().load_state_dict(sd, strict=True)


def test_non_strict_ignores_extras():
    model = make_model()
    sd = model.state_dict()
    sd["bogus.param"] = np.zeros(3, dtype=np.float32)
    # Should not raise; extras ignored.
    make_model().load_state_dict(sd, strict=False)


def test_shape_mismatch_raises():
    model = make_model()
    sd = model.state_dict()
    first = next(iter(sd))
    sd[first] = np.zeros((99, 99), dtype=np.float32)
    with pytest.raises(ValueError):
        make_model().load_state_dict(sd)


def test_save_accepts_raw_state_dict(tmp_path):
    model = make_model()
    path = tmp_path / "raw.npz"
    mg.save(model.state_dict(), path)
    loaded = mg.load(path)
    for k, v in model.state_dict().items():
        assert np.allclose(loaded[k], v)


def test_loaded_weights_train(tmp_path):
    # A loaded model must be fully functional (forward + backward).
    model = make_model()
    path = tmp_path / "m.npz"
    mg.save(model, path)
    fresh = make_model().load(path)
    x = mg.Tensor(np.random.randn(2, 8).astype(np.float32))
    out = fresh(x)
    loss = (out * out).sum()
    loss.backward()
    assert fresh.state_dict()  # params exist and grad flowed without error
