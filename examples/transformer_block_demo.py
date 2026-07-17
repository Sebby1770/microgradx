"""
TransformerEncoderLayer demo — train one block on random (N, S, D) data
for a few steps (identity regression + tiny CE head).

Run:    python examples/transformer_block_demo.py
Args:   --steps N  --batch-size B  --d-model D  --nhead H
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import microgradx as mg
from microgradx import nn, optim, EarlyStopping, EMA
from microgradx.utils import count_parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    mg.manual_seed(0)
    D, H, S, B = args.d_model, args.nhead, args.seq_len, args.batch_size
    n_classes = 4

    class BlockClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.TransformerEncoderLayer(
                d_model=D, nhead=H, dim_feedforward=D * 4, dropout=0.1
            )
            self.head = nn.Linear(D, n_classes)

        def forward(self, x):
            # x: (N, S, D) → pool over sequence → logits (N, C)
            h = self.block(x)          # (N, S, D)
            pooled = h.mean(axis=1)    # (N, D)
            return self.head(pooled)

    model = BlockClassifier()
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=0.99)
    es = EarlyStopping(patience=8, mode="min")
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    print(f"params: {count_parameters(model):,}")
    print(f"TransformerEncoderLayer d_model={D} nhead={H}")

    rng = np.random.default_rng(0)
    for step in range(1, args.steps + 1):
        model.train()
        # Synthetic: class = which channel of mean is largest (toy signal)
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
        ema.update()

        val = float(loss.data)
        print(f"step {step:02d}/{args.steps}  loss={val:.4f}")
        if es.step(val):
            print(f"early stop at step {step} (best={es.best:.4f})")
            break

    model.eval()
    with mg.no_grad(), ema.average_parameters():
        x_eval = mg.Tensor(rng.normal(size=(B, S, D)).astype(np.float32))
        out = model(x_eval)
        print(f"eval (EMA weights) logits shape: {out.shape}")
    print("done.")


if __name__ == "__main__":
    main()
