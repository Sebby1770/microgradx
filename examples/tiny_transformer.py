"""
Tiny Transformer LM — character-level next-token prediction on a synthetic
"add two numbers" toy corpus. The model must learn to attend to past digits
and emit the answer.

Examples in the corpus look like:
    "37+58=95;"
    "12+04=16;"

The model is trained left-to-right with a causal mask. At inference time
we feed it a prompt like "47+25=" and let it autoregressively decode digits
until it emits ';'.

Run:    python examples/tiny_transformer.py
Args:   --epochs N   --batch-size B   --d-model D   --n-layers L
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

import microgradx as mg
from microgradx import nn, optim
from microgradx.data import TensorDataset, DataLoader


# ----- toy corpus -----
VOCAB = "0123456789+=;_"  # _ = pad
PAD = VOCAB.index("_")
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for c, i in STOI.items()}
VOCAB_SIZE = len(VOCAB)
SEQ_LEN = 10  # enough for "AA+BB=CCC;_" (max 10 chars)


def encode(s):
    """str → int array of length SEQ_LEN, right-padded."""
    ids = [STOI[c] for c in s]
    ids = ids + [PAD] * (SEQ_LEN - len(ids))
    return np.array(ids, dtype=np.int64)


def gen_sample(rng):
    a, b = rng.integers(0, 100), rng.integers(0, 100)
    s = f"{a:02d}+{b:02d}={a+b:03d};"
    return s


def make_dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    X = np.stack([encode(gen_sample(rng)) for _ in range(n)])
    # Targets are just X shifted left by one (next-token prediction)
    Y = np.concatenate([X[:, 1:], np.full((len(X), 1), PAD, dtype=np.int64)], axis=1)
    return X, Y


# ----- model -----
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ff_mult * d_model)
        self.fc2 = nn.Linear(ff_mult * d_model, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), causal=True)
        h = self.fc2(self.act(self.fc1(self.ln2(x))))
        x = x + self.drop(h)
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=64, n_heads=4, n_layers=2,
                 dropout=0.0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.seq_len = seq_len

    def forward(self, idx):
        # idx: (B, T) int64
        if isinstance(idx, mg.Tensor):
            B, T = idx.shape
        else:
            B, T = idx.shape
        positions = np.arange(T, dtype=np.int64)
        x = self.token_emb(idx) + self.pos_emb(positions)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, V)


def loss_fn(logits, targets):
    """Flatten across batch+time, ignore PAD tokens (set to weight 0)."""
    B, T, V = logits.shape
    flat_logits = logits.reshape(B * T, V)
    flat_targets = targets if isinstance(targets, np.ndarray) else targets.numpy()
    flat_targets = flat_targets.reshape(B * T)
    # Mask: don't penalise predictions ON pad positions
    mask = (flat_targets != PAD)
    if mask.sum() == 0:
        return mg.Tensor(0.0)
    # Use slicing to drop pad entries
    keep = np.where(mask)[0]
    sub_logits = flat_logits[keep]
    sub_targets = flat_targets[keep]
    return nn.cross_entropy(sub_logits, sub_targets)


def generate(model, prompt, max_new=10):
    """Greedy autoregressive decoding starting from `prompt` (a string)."""
    model.eval()
    ids = list(encode(prompt))[:len(prompt)]
    for _ in range(max_new):
        # pad to SEQ_LEN
        cur = ids + [PAD] * (SEQ_LEN - len(ids))
        x = mg.Tensor(np.array([cur], dtype=np.int64))
        logits = model(x)  # (1, SEQ_LEN, V)
        next_id = int(np.argmax(logits.numpy()[0, len(ids) - 1]))
        ids.append(next_id)
        if next_id == STOI[";"]:
            break
        if len(ids) >= SEQ_LEN:
            break
    return "".join(ITOS[i] for i in ids if i != PAD)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-train", type=int, default=4000)
    args = parser.parse_args()

    Xtr, Ytr = make_dataset(args.n_train, seed=0)
    Xte, Yte = make_dataset(500, seed=1)
    print(f"train: {Xtr.shape}, vocab={VOCAB_SIZE}")
    print(f"sample input : {''.join(ITOS[i] for i in Xtr[0])}")

    train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=args.batch_size,
                               shuffle=True, drop_last=True)

    model = TinyTransformer(VOCAB_SIZE, SEQ_LEN,
                              d_model=args.d_model, n_heads=args.n_heads,
                              n_layers=args.n_layers)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    n_params = sum(p.size for p in model.parameters())
    print(f"model params: {n_params:,}")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batch = 0
        for xb, yb in train_loader:
            x = mg.Tensor(xb)
            logits = model(x)
            loss = loss_fn(logits, yb)
            model.zero_grad()
            loss.backward()
            optim.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += float(loss.data)
            n_batch += 1

        # quick test-set accuracy on next-token at "=" position only
        model.eval()
        correct = 0
        for i in range(len(Xte)):
            s = "".join(ITOS[c] for c in Xte[i] if c != PAD)
            prompt = s.split("=")[0] + "="
            answer = s.split("=")[1].rstrip(";")
            pred = generate(model, prompt, max_new=4).split("=")[-1].rstrip(";")
            if pred == answer:
                correct += 1
        acc = correct / len(Xte)
        print(f"epoch {epoch+1}/{args.epochs}  "
              f"loss={running_loss/n_batch:.4f}  "
              f"exact-match={acc*100:.1f}%  "
              f"({time.time()-t0:.1f}s)")

    print("\nSample generations:")
    for prompt in ["47+25=", "01+09=", "99+99="]:
        print(f"  {prompt} -> {generate(model, prompt)}")


if __name__ == "__main__":
    main()
