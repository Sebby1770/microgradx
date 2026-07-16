"""Adam (Kingma & Ba, 2015) — adaptive moments with L2 weight decay on grads.

For each step t with grad gₜ on parameter θ:

    g'ₜ = gₜ + wd · θ                 # classic L2 (coupled into the grad)
    mₜ = β₁·mₜ₋₁ + (1-β₁)·g'ₜ
    vₜ = β₂·vₜ₋₁ + (1-β₂)·(g'ₜ)²
    m̂ₜ = mₜ / (1 - β₁ᵗ)                # bias correction
    v̂ₜ = vₜ / (1 - β₂ᵗ)
    θ ← θ - lr · m̂ₜ / (√v̂ₜ + ε)

Contrast with AdamW, which applies weight decay *outside* the adaptive step
(decoupled: θ ← θ - lr·(adaptive + wd·θ)).
"""
import numpy as np
from microgradx.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                       weight_decay=weight_decay))

    def step(self):
        self.step_count += 1
        t = self.step_count
        lr = self.defaults["lr"]
        beta1, beta2 = self.defaults["betas"]
        eps = self.defaults["eps"]
        wd = self.defaults["weight_decay"]

        bc1 = 1 - beta1 ** t
        bc2 = 1 - beta2 ** t

        for p, st in zip(self.params, self.state):
            if p.grad is None:
                continue
            g = p.grad
            # Classic L2: fold weight decay into the gradient before moments.
            if wd != 0:
                g = g + wd * p.data

            if "m" not in st:
                st["m"] = np.zeros_like(p.data)
                st["v"] = np.zeros_like(p.data)
            m, v = st["m"], st["v"]
            m *= beta1
            m += (1 - beta1) * g
            v *= beta2
            v += (1 - beta2) * (g * g)

            m_hat = m / bc1
            v_hat = v / bc2

            p.data = p.data - lr * (m_hat / (np.sqrt(v_hat) + eps))
