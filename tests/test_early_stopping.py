"""EarlyStopping helper."""
import microgradx as mg
from microgradx.training import EarlyStopping


def test_early_stopping_stops_after_patience():
    es = EarlyStopping(patience=3, mode="min")
    # Improving then flat
    assert es.step(1.0) is False
    assert es.step(0.9) is False  # improve
    assert es.step(0.95) is False  # worse, counter=1
    assert es.step(0.96) is False  # counter=2
    assert es.step(0.97) is True   # counter=3 → stop
    assert es.stopped is True


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(patience=2, mode="min")
    assert es.step(1.0) is False
    assert es.step(1.1) is False  # counter=1
    assert es.step(0.5) is False  # improve, counter=0
    assert es.step(0.6) is False  # counter=1
    assert es.step(0.7) is True   # counter=2 → stop


def test_early_stopping_mode_max():
    es = EarlyStopping(patience=2, mode="max")
    assert es.step(0.5) is False
    assert es.step(0.4) is False  # worse
    assert es.step(0.3) is True   # stop


def test_early_stopping_top_level_export():
    es = mg.EarlyStopping(patience=1, mode="min")
    assert es.step(1.0) is False
    assert es.step(1.0) is True
