"""
TransformerEncoder + PositionalEncoding demo — train a small encoder stack
on synthetic sequence classification data for a few steps.

Run:    python examples/encoder_stack_demo.py
Args:   --steps N  --batch-size B  --d-model D  --nhead H  --layers L
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import microgradx as mg
from microgradx import nn, optim, accuracy
from microgradx.utils import count_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    mg.manual_seed(0)
    D, H, S, B, L = args.d_model, args.nhead, args.seq_len, args.batch_size, args.layers
    n_classes = 4

    class EncoderClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.pe = nn.PositionalEncoding(D, max_len=S + 8, dropout=0.1)
            self.encoder = nn.TransformerEncoder.from_config(
                d_model=D,
                nhead=H,
                num_layers=L,
                dim_feedforward=D * 4,
                dropout=0.1,
                final_norm=True,
            )
            self.head = nn.Linear(D, n_classes)

        def forward(self, x):
            # x: (N, S, D)
            h = self.encoder(self.pe(x))
            pooled = h.mean(axis=1)
            return self.head(pooled)

    model = EncoderClassifier()
    opt = optim.RAdam(model.parameters(), lr=args.lr)
    sched = optim.CosineAnnealingWarmRestarts(opt, T_0=8, T_mult=1, eta_min=1e-5)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    print(f"params: {count_parameters(model):,}")
    print(f"TransformerEncoder x{L} + PE  d_model={D} nhead={H}")

    rng = np.random.default_rng(0)
    for step in range(1, args.steps + 1):
        model.train()
        x_np = rng.normal(size=(B, S, D)).astype(np.float32) * 0.5
        y_np = rng.integers(0, n_classes, size=B).astype(np.int64)
        for i in range(B):
            x_np[i, :, y_np[i] % D] += 1.0

        x = mg.Tensor(x_np)
        logits = model(x)
        loss = crit(logits, y_np)
        model.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        acc = accuracy(logits, y_np, topk=1)
        print(
            f"step {step:02d}/{args.steps}  loss={float(loss.data):.4f}  "
            f"acc={acc:.2f}  lr={opt.defaults['lr']:.2e}"
        )

    model.eval()
    with mg.no_grad():
        x_eval = mg.Tensor(rng.normal(size=(B, S, D)).astype(np.float32))
        out = model(x_eval)
        print(f"eval logits shape: {out.shape}")
    print("done.")


if __name__ == "__main__":
    main()
