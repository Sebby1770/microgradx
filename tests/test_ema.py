"""EMA of parameters — update + context restores weights."""
import numpy as np
import microgradx as mg
from microgradx import nn
from microgradx.training import EMA


def test_ema_update_and_context_restores():
    model = nn.Linear(4, 2)
    # Snapshot original weights
    orig = {n: p.numpy().copy() for n, p in model.named_parameters()}

    ema = EMA(model, decay=0.5)
    # Mutate parameters
    for p in model.parameters():
        p.data = p.data + 1.0

    mutated = {n: p.numpy().copy() for n, p in model.named_parameters()}
    ema.update()

    # Shadow should be 0.5 * orig + 0.5 * mutated
    for name, p in model.named_parameters():
        expected = 0.5 * orig[name] + 0.5 * mutated[name]
        np.testing.assert_allclose(ema.shadow[name], expected, atol=1e-6)

    # Context manager swaps in shadow then restores
    with ema.average_parameters():
        for name, p in model.named_parameters():
            np.testing.assert_allclose(p.numpy(), ema.shadow[name], atol=1e-6)

    # After exit, original (mutated) weights restored
    for name, p in model.named_parameters():
        np.testing.assert_allclose(p.numpy(), mutated[name], atol=1e-6)


def test_ema_top_level_export():
    model = nn.Linear(2, 1)
    ema = mg.EMA(model, decay=0.9)
    ema.update()
    with ema.average_parameters():
        pass


def test_ema_multiple_updates():
    model = nn.Linear(3, 1, bias=False)
    for p in model.parameters():
        p.data = np.zeros_like(p.data)
    ema = EMA(model, decay=0.9)
    for p in model.parameters():
        p.data = np.ones_like(p.data)
    ema.update()
    # shadow = 0.9 * 0 + 0.1 * 1 = 0.1
    for name in ema.shadow:
        np.testing.assert_allclose(ema.shadow[name], 0.1, atol=1e-6)
