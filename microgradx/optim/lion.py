"""Lion (EvoLved Sign Momentum) — Chen et al., 2023.

Drop-in replacement for AdamW that needs roughly half the memory (one
state buffer instead of two) and often matches/beats it at scale.

    cₜ = β₁·mₜ₋₁ + (1-β₁)·gₜ          # interpolated momentum
    θ ← θ - lr·(sign(cₜ) + wd·θ)       # sign update + decoupled WD
    mₜ = β₂·mₜ₋₁ + (1-β₂)·gₜ          # slower momentum buffer
"""
import numpy as np
from microgradx.optim.optimizer import Optimizer


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    def step(self):
        self.step_count += 1
        lr = self.defaults["lr"]
        beta1, beta2 = self.defaults["betas"]
        wd = self.defaults["weight_decay"]

        for p, st in zip(self.params, self.state):
            if p.grad is None:
                continue
            g = p.grad

            if "m" not in st:
                st["m"] = np.zeros_like(p.data)
            m = st["m"]

            # Update direction = sign(β₁·m + (1-β₁)·g)
            c = beta1 * m + (1 - beta1) * g
            update = np.sign(c)
            if wd != 0:
                update = update + wd * p.data
            p.data = p.data - lr * update

            # Update momentum buffer with the slower beta2
            m *= beta2
            m += (1 - beta2) * g
