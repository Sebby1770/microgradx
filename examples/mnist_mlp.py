"""
MNIST MLP — full end-to-end training using the Trainer.

Pulls MNIST from the OpenML mirror via sklearn if available, otherwise
falls back to a tiny synthetic dataset so the script runs anywhere.

Run:    python examples/mnist_mlp.py
Args:   --epochs N   --batch-size B   --lr LR
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

import microgradx as mg
from microgradx import nn, optim
from microgradx.data import TensorDataset, DataLoader
from microgradx.training import Trainer


def load_mnist():
    """Try the OpenML cache; fall back to a synthetic toy if unavailable."""
    try:
        from sklearn.datasets import fetch_openml
        print("[data] fetching MNIST from openml…")
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, cache=True,
                             return_X_y=True)
        X = X.astype(np.float32) / 255.0
        y = y.astype(np.int64)
        # Standard split: 60k train, 10k test
        return (X[:60000], y[:60000]), (X[60000:], y[60000:])
    except Exception as e:
        print(f"[data] sklearn unavailable ({e}); using 5000-sample synthetic stand-in")
        rng = np.random.default_rng(0)
        # Synthetic: 10 well-separated Gaussian clusters in 784-D
        centers = rng.normal(scale=2, size=(10, 784)).astype(np.float32)
        n_per = 500
        X = np.concatenate([centers[c] + 0.3 * rng.normal(size=(n_per, 784)).astype(np.float32)
                             for c in range(10)])
        y = np.repeat(np.arange(10, dtype=np.int64), n_per)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        split = int(0.85 * len(X))
        return (X[:split], y[:split]), (X[split:], y[split:])


class MLP(nn.Module):
    """784 → 256 → 128 → 10."""
    def __init__(self, in_dim=784, hidden=(256, 128), out_dim=10, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.fc1(x).relu())
        x = self.dropout(self.fc2(x).relu())
        return self.fc3(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    args = parser.parse_args()

    (Xtr, ytr), (Xte, yte) = load_mnist()
    print(f"train: {Xtr.shape}, test: {Xte.shape}, dtype: {Xtr.dtype}")

    train_loader = DataLoader(TensorDataset(Xtr, ytr),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(Xte, yte),
                             batch_size=512, shuffle=False)

    model = MLP()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    trainer = Trainer(model, opt, loss_fn=nn.cross_entropy,
                       grad_clip=args.grad_clip, log_every=100)

    for epoch in range(args.epochs):
        t0 = time.time()
        train_stats = trainer.train_epoch(train_loader)
        eval_stats = trainer.evaluate(test_loader)
        dt = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs}  "
              f"train_loss={train_stats['loss']:.4f}  "
              f"train_acc={train_stats['accuracy']*100:.2f}%  "
              f"test_loss={eval_stats['loss']:.4f}  "
              f"test_acc={eval_stats['accuracy']*100:.2f}%  "
              f"({dt:.1f}s)")


if __name__ == "__main__":
    main()
