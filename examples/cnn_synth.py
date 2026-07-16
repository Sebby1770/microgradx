"""
Tiny CNN on synthetic (N, 3, 32, 32) images.

Architecture: Conv2d → GroupNorm → SiLU → Conv2d → GroupNorm → SiLU
            → AdaptiveAvgPool2d(1) → Flatten → Linear.

Trains for a few steps so the example finishes quickly without real data.

Run:    python examples/cnn_synth.py
Args:   --steps N   --batch-size B   --lr LR
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import microgradx as mg
from microgradx import nn, optim, CSVLogger
from microgradx.utils import count_parameters, summary


def make_data(n=256, n_classes=4, seed=0):
    """Synthetic RGB 32×32: class c has elevated mean on channel c % 3."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3, 32, 32)).astype(np.float32) * 0.3
    y = rng.integers(0, n_classes, size=n).astype(np.int64)
    for i in range(n):
        X[i, y[i] % 3] += 0.8
    return X, y


class TinyCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 16)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(8, 32)
        self.act2 = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout2d(0.1)
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.act1(self.gn1(self.conv1(x)))
        x = self.drop(self.act2(self.gn2(self.conv2(x))))
        x = self.pool(x)
        x = nn.Flatten()(x)
        return self.fc(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--log", type=str, default="")
    args = parser.parse_args()

    X, y = make_data()
    model = TinyCNN()
    print(summary(model, input_shape=(args.batch_size, 3, 32, 32)))
    print(f"parameters: {count_parameters(model):,}")

    opt = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    logger = CSVLogger(args.log) if args.log else None
    rng = np.random.default_rng(1)
    model.train()
    losses = []
    for step in range(1, args.steps + 1):
        idx = rng.choice(len(X), size=args.batch_size, replace=False)
        xb, yb = X[idx], y[idx]
        logits = model(mg.Tensor(xb))
        loss = nn.cross_entropy(logits, yb)
        model.zero_grad()
        loss.backward()
        opt.step()
        lv = float(loss.data)
        losses.append(lv)
        sched.step(lv)
        if logger is not None:
            logger.log(step=step, loss=lv, lr=sched.get_last_lr())
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"step {step:3d}  loss={lv:.4f}  lr={sched.get_last_lr():.5f}")

    if logger is not None:
        logger.close()

    assert losses[-1] < losses[0], "expected training loss to drop"
    # quick eval accuracy
    model.eval()
    with mg.no_grad():
        logits = model(mg.Tensor(X[:64]))
        pred = logits.numpy().argmax(axis=-1)
        acc = float((pred == y[:64]).mean())
    print(f"train-subset acc@64: {acc:.2%}")
    print("done.")


if __name__ == "__main__":
    main()
