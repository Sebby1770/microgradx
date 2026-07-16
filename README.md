# MicroGradX

Minimalist autograd & neural-net framework in pure Python + NumPy.
Functional shape: `Tensor` + dynamic graph + `nn.Module` + recurrent & conv
layers + optimisers + LR schedulers (incl. OneCycle) + INT8 quantisation +
`no_grad` inference + training loop + save/load + ONNX export.

```
import microgradx as mg
from microgradx import nn, optim

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
opt = optim.AdamW(model.parameters(), lr=1e-3)
sched = optim.OneCycleLR(opt, max_lr=1e-2, total_steps=1000)

for x, y in loader:
    logits = model(mg.Tensor(x))
    loss = nn.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()
    opt.step()
    sched.step()

# inference without building the graph; optional INT8 weight quantisation
with mg.no_grad():
    preds = model(mg.Tensor(x_test))
qmodel = mg.quant.quantize_dynamic(model)
print(mg.summary(model), mg.count_parameters(model))
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
│   │   ├── conv.py          # Conv1d, Conv2d, MaxPool2d, Flatten
│   │   ├── rnn.py           # RNN, GRU, LSTM
│   │   ├── norm.py          # LayerNorm, RMSNorm, BatchNorm1d/2d
│   │   ├── dropout.py       # Dropout
│   │   ├── activation.py    # ReLU, GELU, Sigmoid, Tanh, Softmax
│   │   ├── embedding.py     # Embedding
│   │   ├── attention.py     # MultiHeadAttention + SDPA
│   │   ├── loss.py          # CrossEntropy, MSE
│   │   └── init.py          # Kaiming, Xavier, …
│   ├── optim/
│   │   ├── optimizer.py     # base + grad clipping
│   │   ├── sgd.py / adamw.py / lion.py
│   │   └── lr_scheduler.py  # Step, Cosine, OneCycle, …
│   ├── quant/               # dynamic INT8 Linear quantisation
│   ├── data/                # Dataset, DataLoader, transforms
│   ├── training/            # Trainer, AMP plumbing
│   ├── export/              # ONNX exporter
│   └── utils.py             # checkpoint, count_parameters, summary
├── tests/
├── examples/
│   ├── mnist_mlp.py
│   ├── tiny_transformer.py
│   └── seq_classify.py      # GRU sequence classification
├── docs/
│   ├── MATH.md
│   └── ROADMAP.md
└── README.md
```

---

## What's actually in here

| | |
|---|---|
| **Autograd** | dynamic DAG, iterative topological sort, broadcasting via `_unbroadcast` |
| **Custom ops** | subclass `Function`, override `forward`/`backward`, validate with `gradcheck` |
| **Layers** | Linear, **Conv1d**, Conv2d (dual-path im2col), MaxPool2d, LayerNorm, RMSNorm, BatchNorm1d/2d, Dropout, Embedding, MultiHeadAttention, **RNN / GRU / LSTM** |
| **Inference** | `no_grad()` / `enable_grad()`; **dynamic INT8** via `mg.quant.quantize_dynamic` |
| **Memory** | `mg.checkpoint(fn, *args)` activation checkpointing |
| **Optimisers** | SGD, AdamW, Lion + L∞ / L2 gradient clipping |
| **Schedulers** | StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, LinearWarmup, LambdaLR, **OneCycleLR** |
| **Utils** | `count_parameters`, `summary` |
| **Training** | DataLoader, augmentation, grad accumulation, mixed-precision plumbing |
| **Persistence** | `mg.save` / `mg.load` to portable `.npz`, `Module.state_dict` / `load_state_dict` |
| **Export** | trace → ONNX (or JSON if onnx not installed) |

---

## Recurrent layers

```python
rnn = nn.RNN(input_size=16, hidden_size=32, num_layers=2, batch_first=True)
gru = nn.GRU(16, 32, batch_first=True)
lstm = nn.LSTM(16, 32, num_layers=1, batch_first=True)

x = mg.randn(4, 10, 16)          # (batch, seq, features)
out, h_n = gru(x)                # out: (4, 10, 32), h_n: (1, 4, 32)
out, (h_n, c_n) = lstm(x)
```

---

## OneCycleLR

```python
opt = optim.AdamW(model.parameters(), lr=1e-3)
sched = optim.OneCycleLR(opt, max_lr=1e-2, total_steps=len(loader) * epochs,
                         pct_start=0.3, anneal_strategy="cos")
for epoch in range(epochs):
    for batch in loader:
        ...
        opt.step()
        sched.step()   # once per batch
```

---

## Dynamic INT8 quantisation

```python
from microgradx.quant import quantize_dynamic

model.eval()
qmodel = quantize_dynamic(model)   # Linear → Int8Linear (weight absmax scale)
with mg.no_grad():
    y = qmodel(mg.Tensor(x))
```

Weights are stored as int8 + a per-tensor scale and dequantised to float32
for the matmul — correct and simple for a teaching framework.

---

## Run the tests

```bash
cd microgradx
python3 -m pytest tests/ -q
```

85 tests, well under a second. Includes `gradcheck` for primitives,
RNN/GRU/LSTM shapes + grad flow, Conv1d, OneCycleLR, INT8 quant,
`no_grad`, schedulers, save/load, BatchNorm, and checkpointing.

## Run the examples

```bash
python3 examples/mnist_mlp.py --epochs 3
python3 examples/tiny_transformer.py --epochs 8
python3 examples/seq_classify.py --epochs 12
```

The MNIST loader uses sklearn's OpenML cache if available; otherwise a
synthetic stand-in. `seq_classify.py` trains a tiny GRU on synthetic
sequences and ends with an INT8 quantised eval.

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

x = Tensor(np.random.randn(3, 4), requires_grad=True)
assert gradcheck(lambda x: Swish.apply(x).sum(), [x])
```

## Read the math

See [docs/MATH.md](docs/MATH.md) for closed-form gradients for every op,
including the GELU `tanh` approximation, softmax/log-softmax stable
formulations, and the im2col/col2im pair behind Conv1d/Conv2d.

## What's next

See [docs/ROADMAP.md](docs/ROADMAP.md) — CuPy/AMP, Ray-based DDP,
QAT / 4-bit weight-only quantisation.

## License

MIT — see [LICENSE](LICENSE).
