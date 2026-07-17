"""RAdam — Rectified Adam (Liu et al., ICLR 2020).

Rectifies the adaptive learning-rate term so early-step variance estimates
don't produce unstable updates. When the approximated SMA length ``ρ_t`` is
large enough (``ρ_t > 4``), use a variance-adapted step; otherwise fall back
to bias-corrected momentum only (no adaptive denominator).

    ρ∞ = 2/(1-β₂) − 1
    ρₜ = ρ∞ − 2·t·β₂ᵗ / (1 − β₂ᵗ)
    m̂ₜ = mₜ / (1 − β₁ᵗ)

    if ρₜ > 4:
        rₜ = √[ ((ρₜ−4)(ρₜ−2)ρ∞) / ((ρ∞−4)(ρ∞−2)ρₜ) ]
        v̂ₜ = vₜ / (1 − β₂ᵗ)
        θ ← θ − lr · rₜ · m̂ₜ / (√v̂ₜ + ε)
    else:
        θ ← θ − lr · m̂ₜ
"""
from __future__ import annotations

import math

import numpy as np

from microgradx.optim.optimizer import Optimizer


class RAdam(Optimizer):
    """Rectified Adam optimiser.

    Parameters
    ----------
    params : iterable of Tensor
    lr : float
    betas : (float, float)
        Coefficients used for computing running averages of gradient and its square.
    eps : float
        Term added to the denominator for numerical stability.
    weight_decay : float
        Classic L2 weight decay folded into the gradient (Adam-style).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )

    def step(self):
        self.step_count += 1
        t = self.step_count
        lr = self.defaults["lr"]
        beta1, beta2 = self.defaults["betas"]
        eps = self.defaults["eps"]
        wd = self.defaults["weight_decay"]

        # ρ∞ = 2/(1-β₂) − 1
        rho_inf = 2.0 / (1.0 - beta2) - 1.0
        beta2_t = beta2 ** t
        # ρₜ = ρ∞ − 2 t β₂ᵗ / (1 − β₂ᵗ)
        rho_t = rho_inf - 2.0 * t * beta2_t / (1.0 - beta2_t)

        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2_t

        # Rectification term (only used when ρₜ is reliable)
        if rho_t > 4.0:
            r_t = math.sqrt(
                ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf)
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
            )
        else:
            r_t = None

        for p, st in zip(self.params, self.state):
            if p.grad is None:
                continue
            g = p.grad
            if wd != 0:
                g = g + wd * p.data

            if "m" not in st:
                st["m"] = np.zeros_like(p.data)
                st["v"] = np.zeros_like(p.data)
            m, v = st["m"], st["v"]
            m *= beta1
            m += (1.0 - beta1) * g
            v *= beta2
            v += (1.0 - beta2) * (g * g)

            m_hat = m / bc1

            if r_t is not None:
                v_hat = v / bc2
                update = r_t * m_hat / (np.sqrt(v_hat) + eps)
            else:
                # Variance not yet reliable — plain bias-corrected momentum.
                update = m_hat

            p.data = p.data - lr * update
