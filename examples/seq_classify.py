"""
Tiny GRU sequence classifier on synthetic data.

Each sequence is length T of F-dim features. Class 0 sequences have a
slightly higher mean in the first half; class 1 in the second half.
A small GRU + Linear head learns to tell them apart.

Run:    python examples/seq_classify.py
Args:   --epochs N   --batch-size B   --lr LR
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import microgradx as mg
from microgradx import nn, optim
from microgradx.data import TensorDataset, DataLoader
from microgradx.utils import count_parameters, summary


def make_data(n=800, T=16, F=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, T, F)).astype(np.float32) * 0.3
    y = rng.integers(0, 2, size=n).astype(np.int64)
    half = T // 2
    for i in range(n):
        if y[i] == 0:
            X[i, :half, :] += 0.6
        else:
            X[i, half:, :] += 0.6
    return X, y


class SeqClassifier(nn.Module):
    def __init__(self, input_size=8, hidden=32, n_classes=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x: (B, T, F)
        out, h_n = self.gru(x)
        # last layer final hidden: (1, B, H) → (B, H)
        h = h_n[0]
        return self.fc(h)


def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with mg.no_grad():
        for xb, yb in loader:
            logits = model(mg.Tensor(xb))
            pred = logits.numpy().argmax(axis=-1)
            correct += int((pred == np.asarray(yb)).sum())
            total += len(yb)
    model.train()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    X, y = make_data()
    split = int(0.8 * len(X))
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(Xte, yte), batch_size=64, shuffle=False
    )

    model = SeqClassifier()
    print(summary(model, input_shape=(args.batch_size, 16, 8)))
    print(f"parameters: {count_parameters(model):,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * max(1, len(Xtr) // args.batch_size)
    sched = optim.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            logits = model(mg.Tensor(xb))
            loss = nn.cross_entropy(logits, yb)
            model.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * len(yb)
            n += len(yb)
        acc = accuracy(model, test_loader)
        print(
            f"epoch {epoch:02d}  loss={total_loss / n:.4f}  "
            f"test_acc={acc:.3f}  lr={sched.get_last_lr():.5f}"
        )

    # Optional: quantise for inference and re-check accuracy.
    qmodel = mg.quant.quantize_dynamic(model)
    q_acc = accuracy(qmodel, test_loader)
    print(f"INT8 dynamic quant test_acc={q_acc:.3f}")


if __name__ == "__main__":
    main()
