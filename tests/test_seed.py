"""manual_seed reproducibility."""
import numpy as np
import microgradx as mg


def test_manual_seed_reproducibility_randn():
    mg.manual_seed(123)
    a = mg.randn(4, 4).numpy().copy()
    mg.manual_seed(123)
    b = mg.randn(4, 4).numpy().copy()
    np.testing.assert_array_equal(a, b)


def test_manual_seed_different_seeds_differ():
    mg.manual_seed(1)
    a = mg.randn(8).numpy().copy()
    mg.manual_seed(2)
    b = mg.randn(8).numpy().copy()
    assert not np.allclose(a, b)


def test_manual_seed_exported():
    assert hasattr(mg, "manual_seed")
    assert callable(mg.manual_seed)
