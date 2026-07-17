# MicroGradX

Minimalist autograd & neural-net framework in pure Python + NumPy.
Functional shape: `Tensor` + dynamic graph + `nn.Module` + CNN / recurrent /
transformer encoder stack + optimisers (Adam / AdamW / RAdam) + LR schedulers
(warm restarts) + EMA / early stopping + metrics + INT8 quantisation +
`no_grad` inference + training loop + CSV logging + save/load + ONNX export.

```
import microgradx as mg
from microgradx import nn, optim, CSVLogger, EarlyStopping, EMA

mg.manual_seed(0)
model = nn.Sequential(nn.Linear(784, 128), nn.SiLU(), nn.Linear(128, 10))
opt = optim.Adam(model.parameters(), lr=1e-3)
sched = optim.ReduceLROnPlateau(opt, patience=3)
ema = EMA(model, decay=0.999)
es = EarlyStopping(patience=5, mode="min")
log = CSVLogger("run.csv")

for epoch in range(epochs):
    for x, y in loader:
        logits = model(mg.Tensor(x))
        loss = nn.cross_entropy(logits, y, label_smoothing=0.1)
        model.zero_grad()
        loss.backward()
        optim.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update()
    with ema.average_parameters():
        val = evaluate(model)
    sched.step(val)
    log.log(epoch=epoch, loss=float(loss.data), val=val)
    if es.step(val):
        break
log.close()

with mg.no_grad():
    preds = model(mg.Tensor(x_test))
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
│   │   ├── module.py        # Module, Sequential, ModuleList (+ apply)
│   │   ├── linear.py        # Linear
│   │   ├── conv.py          # Conv1d/2d, Max/Avg/AdaptiveAvgPool2d, Flatten
│   │   ├── rnn.py           # RNN, GRU, LSTM
│   │   ├── norm.py          # LayerNorm, RMSNorm, BatchNorm, GroupNorm
│   │   ├── dropout.py       # Dropout, Dropout2d
│   │   ├── activation.py    # ReLU, LeakyReLU, SiLU, Softplus, GELU, …
│   │   ├── embedding.py     # Embedding
│   │   ├── attention.py     # MultiHeadAttention + SDPA
│   │   ├── transformer.py   # TransformerEncoder, EncoderLayer, PositionalEncoding
│   │   ├── upsample.py      # Upsample / interpolate (nearest + bilinear)
│   │   ├── loss.py          # CrossEntropy, BCE, MSE, Huber / SmoothL1
│   │   └── init.py          # Kaiming, Xavier, …
│   ├── optim/
│   │   ├── optimizer.py     # base + grad clipping
│   │   ├── sgd / adam / adamw / lion / radam
│   │   └── lr_scheduler.py  # Cosine warm restarts, OneCycle, Plateau, …
│   ├── quant/               # dynamic INT8 Linear quantisation
│   ├── data/                # Dataset, DataLoader, transforms
│   ├── training/            # Trainer, AMP, EarlyStopping, EMA
│   ├── export/              # ONNX exporter
│   ├── metrics.py           # top-k accuracy
│   ├── logging.py           # CSVLogger
│   └── utils.py             # checkpoint, count_parameters, summary, manual_seed
├── tests/
├── examples/
│   ├── mnist_mlp.py
│   ├── tiny_transformer.py
│   ├── transformer_block_demo.py  # one TransformerEncoderLayer
│   ├── encoder_stack_demo.py      # TransformerEncoder + PE stack
│   ├── seq_classify.py      # GRU sequence classification
│   └── cnn_synth.py         # Conv + GroupNorm + AdaptiveAvgPool
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
| **Layers** | Linear, Conv1d/2d, Max/Avg/AdaptiveAvgPool2d, **Upsample** (nearest/bilinear), LayerNorm, RMSNorm, BatchNorm1d/2d, GroupNorm, Dropout/Dropout2d, Embedding, MultiHeadAttention, **TransformerEncoder** / **EncoderLayer**, **PositionalEncoding**, RNN/GRU/LSTM |
| **Activations** | ReLU, LeakyReLU, SiLU, Softplus, GELU, Sigmoid, Tanh, Softmax |
| **Losses** | CrossEntropy (**label_smoothing**), **BCEWithLogits** / **BCE**, MSE, **Huber** / **SmoothL1** |
| **Inference** | `no_grad()` / `enable_grad()`; dynamic INT8 via `mg.quant.quantize_dynamic` |
| **Memory** | `mg.checkpoint(fn, *args)` activation checkpointing |
| **Optimisers** | SGD, Adam, AdamW, Lion, **RAdam** + L∞ / L2 gradient clipping |
| **Schedulers** | StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, **CosineAnnealingWarmRestarts**, LinearWarmup, LambdaLR, OneCycleLR, ReduceLROnPlateau |
| **Training helpers** | **EarlyStopping**, **EMA**, CSVLogger, Trainer, AMP plumbing, **`Module.freeze`/`unfreeze`** |
| **Metrics** | **`mg.accuracy`** top-k classification accuracy |
| **Utils** | `count_parameters`, `summary`, **`manual_seed`**, `Module.apply` |
| **Data** | DataLoader, augmentation, grad accumulation |
| **Persistence** | `mg.save` / `mg.load` to portable `.npz`, `Module.state_dict` / `load_state_dict` |
| **Export** | trace → ONNX (or JSON if onnx not installed) |

---

## Encoder stack + PE (v0.6)

```python
enc = nn.TransformerEncoder.from_config(d_model=64, nhead=4, num_layers=4)
pe = nn.PositionalEncoding(64, max_len=512, dropout=0.1)
x = mg.randn(2, 16, 64)          # (B, T, D)
y = enc(pe(x))                    # same shape

opt = optim.RAdam(model.parameters(), lr=1e-3)
sched = optim.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-5)

loss = nn.HuberLoss(delta=1.0)(pred, target)
acc = mg.accuracy(logits, y, topk=(1, 5))

model.freeze()                    # requires_grad=False on all params
head.unfreeze()                   # train only the head

up = nn.Upsample(scale_factor=2, mode="bilinear")
fmap = up(mg.randn(1, 8, 16, 16)) # (1, 8, 32, 32)
```

## Transformer + training helpers (v0.5)

```python
layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256)
x = mg.randn(2, 16, 64)          # (N, S, D)
y = layer(x)                      # same shape

up = nn.Upsample(scale_factor=2, mode="nearest")
fmap = up(mg.randn(1, 8, 16, 16)) # (1, 8, 32, 32)

loss = nn.BCEWithLogitsLoss()(logits, targets)
loss = nn.cross_entropy(logits, y, label_smoothing=0.1)

ema = mg.EMA(model, decay=0.999)
ema.update()
with ema.average_parameters():
    evaluate(model)

es = mg.EarlyStopping(patience=5, mode="min")
if es.step(val_loss):
    break
mg.manual_seed(42)
```

---

## CNN pieces (v0.4)

```python
gn = nn.GroupNorm(num_groups=8, num_channels=32)
pool = nn.AdaptiveAvgPool2d(1)          # global average pool
x = mg.randn(4, 32, 16, 16)
y = pool(nn.SiLU()(gn(x)))              # (4, 32, 1, 1)

opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = optim.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
sched.step(val_loss)
```

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

Includes `gradcheck` for primitives, CNN pools / GroupNorm / activations,
Adam + ReduceLROnPlateau, CSVLogger, RNN/GRU/LSTM, Conv1d, OneCycleLR,
INT8 quant, `no_grad`, schedulers, save/load, BatchNorm, and checkpointing.

## Run the examples

```bash
python3 examples/mnist_mlp.py --epochs 3
python3 examples/tiny_transformer.py --epochs 8
python3 examples/seq_classify.py --epochs 12
python3 examples/cnn_synth.py --steps 30
```

The MNIST loader uses sklearn's OpenML cache if available; otherwise a
synthetic stand-in. `seq_classify.py` trains a tiny GRU on synthetic
sequences. `cnn_synth.py` trains a small Conv+GroupNorm net on synthetic
RGB 32×32 images.

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
