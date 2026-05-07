"""SGD with optional momentum, Nesterov, and weight decay."""
import numpy as np
from microgradx.optim.optimizer import Optimizer


class SGD(Optimizer):
    """θ ← θ - lr · (g + wd·θ)
    With momentum:  v ← μ·v + g + wd·θ;  θ ← θ - lr · v       (or Nesterov variant)
    """
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                       weight_decay=weight_decay, nesterov=nesterov))
        if nesterov and momentum == 0:
            raise ValueError("Nesterov requires momentum > 0")

    def step(self):
        self.step_count += 1
        lr = self.defaults["lr"]
        mu = self.defaults["momentum"]
        wd = self.defaults["weight_decay"]
        nesterov = self.defaults["nesterov"]

        for p, st in zip(self.params, self.state):
            if p.grad is None:
                continue
            g = p.grad
            if wd != 0:
                g = g + wd * p.data
            if mu != 0:
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = np.zeros_like(p.data)
                buf = st["momentum_buffer"]
                buf *= mu
                buf += g
                if nesterov:
                    g = g + mu * buf
                else:
                    g = buf
            p.data = p.data - lr * g
