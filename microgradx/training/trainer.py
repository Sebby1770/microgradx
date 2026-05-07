"""
Trainer — opinionated loop with gradient accumulation, AMP, grad clipping.

Usage:

    trainer = Trainer(
        model, optimizer, loss_fn=nn.cross_entropy,
        grad_accum_steps=4, grad_clip=1.0, use_amp=False,
    )
    for epoch in range(epochs):
        trainer.train_epoch(train_loader)
        trainer.evaluate(val_loader)
"""
from __future__ import annotations
from typing import Callable, Iterable, Optional, Tuple
import time
import numpy as np

from microgradx.tensor import Tensor
from microgradx.nn.module import Module
from microgradx.optim.optimizer import Optimizer, clip_grad_norm_
from microgradx.training.amp import GradScaler, autocast


class Trainer:
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        grad_accum_steps: int = 1,
        grad_clip: Optional[float] = None,
        use_amp: bool = False,
        log_every: int = 50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.log_every = log_every
        self.scaler = GradScaler(enabled=use_amp)
        self.global_step = 0

    def _step(self, batch) -> Tuple[float, int]:
        x, y = batch
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x))
        if not isinstance(y, Tensor):
            y = Tensor(np.asarray(y))

        with autocast(enabled=self.use_amp):
            logits = self.model(x)
            loss = self.loss_fn(logits, y)

        # Scale for AMP and divide for accumulation, both up front.
        scaled_loss = self.scaler.scale_loss(loss) / self.grad_accum_steps
        scaled_loss.backward()
        # accuracy for logging (only if logits look classification-shaped)
        n_correct = 0
        if logits.ndim == 2:
            preds = np.argmax(logits.data, axis=-1)
            tgt = y.data if isinstance(y, Tensor) else y
            n_correct = int(np.sum(preds == tgt))
        return float(loss.data), n_correct

    def train_epoch(self, loader) -> dict:
        self.model.train()
        t0 = time.time()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        for i, batch in enumerate(loader):
            loss, correct = self._step(batch)
            running_loss += loss
            running_correct += correct
            running_seen += len(batch[1] if isinstance(batch, tuple) else batch)

            if (i + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                if self.grad_clip is not None:
                    clip_grad_norm_(self.model.parameters(), self.grad_clip)
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.log_every and self.global_step % self.log_every == 0:
                    avg_loss = running_loss / (i + 1)
                    acc = running_correct / max(running_seen, 1)
                    print(f"  step {self.global_step:6d} | loss {avg_loss:.4f} | "
                          f"acc {acc*100:.2f}%")

        return {
            "loss": running_loss / max(len(loader), 1),
            "accuracy": running_correct / max(running_seen, 1),
            "time": time.time() - t0,
        }

    def evaluate(self, loader) -> dict:
        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        for batch in loader:
            x, y = batch
            if not isinstance(x, Tensor):
                x = Tensor(np.asarray(x))
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            running_loss += float(loss.data)
            if logits.ndim == 2:
                preds = np.argmax(logits.data, axis=-1)
                tgt = y.data if isinstance(y, Tensor) else y
                running_correct += int(np.sum(preds == tgt))
                running_seen += len(tgt)
        return {
            "loss": running_loss / max(len(loader), 1),
            "accuracy": running_correct / max(running_seen, 1),
        }
