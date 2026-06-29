# MicroGradX

Minimalist autograd & neural-net framework in pure Python + NumPy.
Functional shape: `Tensor` + dynamic graph + `nn.Module` + optimisers +
LR schedulers + `no_grad` inference + training loop + save/load + ONNX
export. ~3,200 LOC, dependency-free at runtime.

```
import microgradx as mg
from microgradx import nn, optim

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
opt = optim.AdamW(model.parameters(), lr=1e-3)
sched = optim.CosineAnnealingLR(opt, T_max=epochs)

for x, y in loader:
    logits = model(mg.Tensor(x))
    loss = nn.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()
    opt.step()
sched.step()

# inference without building the graph; persist weights to disk
with mg.no_grad():
    preds = model(mg.Tensor(x_test))
mg.save(model, "mnist.npz")
```

---

## Project layout

```
microgradx/
├── microgradx/
│   ├── tensor.py            # Tensor + autograd driver
│   ├── backend/             # numpy ↔ cupy abstraction
│   ├── autograd/
│   │   ├── function.py      # base Function/Context
│   │   ├── ops.py           # all primitive forward/backward pairs
│   │   └── grad_check.py    # fp64 numerical gradient checker
│   ├── nn/
│   │   ├── module.py        # Module, Sequential, ModuleList
│   │   ├── linear.py        # Linear
│   │   ├── conv.py          # Conv2d, MaxPool2d, Flatten (im2col impl)
│   │   ├── norm.py          # LayerNorm, RMSNorm
│   │   ├── dropout.py       # Dropout
│   │   ├── activation.py    # ReLU, GELU, Sigmoid, Tanh, Softmax
│   │   ├── embedding.py     # Embedding (lookup table)
│   │   ├── attention.py     # MultiHeadAttention + scaled dot-product
│   │   ├── loss.py          # CrossEntropy (fused), MSE
│   │   └── init.py          # Kaiming, Xavier, etc.
│   ├── optim/
│   │   ├── optimizer.py     # base + grad clipping
│   │   ├── sgd.py           # SGD (+momentum, +Nesterov, +weight decay)
│   │   ├── adamw.py         # AdamW
│   │   └── lion.py          # Lion
│   ├── data/
│   │   ├── dataset.py       # Dataset, TensorDataset
│   │   ├── dataloader.py    # DataLoader
│   │   └── transforms.py    # Normalize, RandomCrop, RandomHorizontalFlip
│   ├── training/
│   │   ├── amp.py           # autocast, GradScaler
│   │   └── trainer.py       # Trainer (grad accumulation + AMP + clipping)
│   └── export/
│       └── onnx.py          # tracing exporter, onnx soft-dep
├── tests/
│   ├── test_tensor.py
│   ├── test_autograd.py     # gradcheck for every primitive
│   ├── test_nn.py
│   └── test_optim.py
├── examples/
│   ├── mnist_mlp.py         # full MLP training loop
│   └── tiny_transformer.py  # decoder-only LM on a toy arithmetic task
├── docs/
│   ├── MATH.md              # gradient derivations for every op
│   └── ROADMAP.md           # CuPy / Ray distributed / quantisation plans
└── README.md
```

---

## What's actually in here

| | |
|---|---|
| **Autograd** | dynamic DAG, iterative topological sort, broadcasting handled centrally via `_unbroadcast` |
| **Custom ops** | subclass `Function`, override `forward`/`backward`, validate with `gradcheck` |
| **Layers** | Linear, Conv2d (dual-path im2col: stride-trick view for large kernels, slice loop for small), MaxPool2d, LayerNorm, RMSNorm, BatchNorm1d/2d (running stats + train/eval), Dropout, Embedding, MultiHeadAttention with causal mask |
| **Inference** | `no_grad()` / `enable_grad()` context managers + decorators that skip graph construction |
| **Memory** | `mg.checkpoint(fn, *args)` activation checkpointing — recompute in backward instead of storing intermediates |
| **Optimisers** | SGD, AdamW, Lion + L∞ / L2 gradient clipping |
| **Schedulers** | StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearWarmup, LambdaLR |
| **Training** | DataLoader, augmentation, grad accumulation, mixed-precision plumbing (loss scaling) |
| **Persistence** | `mg.save` / `mg.load` to portable `.npz` (no pickle), `Module.state_dict` / `load_state_dict` |
| **Export** | trace → ONNX (or JSON if onnx not installed) for the documented op subset |

---

## Run the tests

```bash
cd microgradx
python3 -m pytest tests/ -q
```

54 tests, well under a second. Includes `gradcheck` for every primitive op,
plus coverage for `no_grad`, every LR scheduler, save/load round-trips, and
the dual-path Conv2d.

## Run the examples

```bash
python3 examples/mnist_mlp.py --epochs 3
python3 examples/tiny_transformer.py --epochs 8
```

The MNIST loader will use sklearn's OpenML cache if available; otherwise
it falls back to a synthetic stand-in. The transformer trains a tiny
character-level LM on `"AA+BB=CCC;"` examples.

## Add a custom op

```python
import numpy as np
from microgradx import Function, Tensor, gradcheck

class Swish(Function):
    @staticmethod
    def forward(ctx, x):
        sig = 1.0 / (1.0 + np.exp(-x))
        ctx.save_for_backward(x, sig)
        return x * sig
    @staticmethod
    def backward(ctx, g):
        x, sig = ctx.saved_tensors
        return g * (sig + x * sig * (1 - sig)),

# verify
x = Tensor(np.random.randn(3, 4), requires_grad=True)
assert gradcheck(lambda x: Swish.apply(x).sum(), [x])
```

## Read the math

See [docs/MATH.md](docs/MATH.md) for closed-form gradients for every op,
including the GELU `tanh` approximation, softmax/log-softmax stable
formulations, and the im2col/col2im pair behind Conv2d.

## What's next

See [docs/ROADMAP.md](docs/ROADMAP.md) — CuPy/AMP, Ray-based DDP,
INT8 / 4-bit quantisation, gradient checkpointing.
