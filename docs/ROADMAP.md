# MicroGradX — Roadmap

What's working today vs what's planned. Status: **v0.5.0**.

---

## ✅ Shipped in v0.5

- **BCEWithLogitsLoss / BCELoss** — stable binary CE (logits + probability forms)
- **Label-smoothing CrossEntropy** — `label_smoothing=` on CE loss / functional
- **Upsample / interpolate** — nearest-neighbor NCHW upsample
- **TransformerEncoderLayer** — MHA + GELU FFN + LayerNorm + residual + Dropout
- **EarlyStopping** — `mg.EarlyStopping(patience, mode)` training helper
- **EMA** — exponential moving average of weights with restore context manager
- **`manual_seed(seed)`** — seed Python random + NumPy (+ CuPy if active)
- **Example**: `examples/transformer_block_demo.py`
- Tests for all of the above

---

## ✅ Shipped in v0.4

- **Activations**: LeakyReLU, SiLU (Swish), Softplus
- **Pooling**: AvgPool2d, AdaptiveAvgPool2d (global pool with `output_size=1`)
- **GroupNorm** — channel-group normalisation, batch-size independent
- **Dropout2d** — channel dropout for conv maps
- **Adam** — classic Adam (L2 weight decay on grads; AdamW remains decoupled)
- **ReduceLROnPlateau** — metric-driven LR reduction
- **CSVLogger** — `mg.CSVLogger("run.csv")` metrics logging
- **`Module.apply(fn)`** — recursive module visitor
- **Example**: `examples/cnn_synth.py`
- New tests for all of the above

---

## ✅ Shipped in v0.3

- **RNN / GRU / LSTM** — multi-layer, `batch_first`, dropout between layers,
  optional initial state; PyTorch-ish `(output, h_n)` / `(output, (h_n, c_n))`
- **Conv1d** — im2col → GEMM, `(N, C, L) → (N, C_out, L_out)`
- **OneCycleLR** — warmup to `max_lr` then cosine/linear anneal
- **Dynamic INT8 quantisation** — `mg.quant.quantize_dynamic` replaces
  `Linear` with weight-only `Int8Linear` (absmax scale)
- **`count_parameters` / `summary`** — quick model inspection helpers
- **Example**: `examples/seq_classify.py` (GRU sequence classification)
- MIT LICENSE, expanded README / CHANGELOG / pyproject metadata

Also included from the checkpoint/BatchNorm lineage:
- **Gradient checkpointing** — `mg.checkpoint(fn, *args)`
- **BatchNorm1d / BatchNorm2d** — running stats + train/eval; buffers
  persist through `mg.save` / `mg.load`

---

## ✅ Shipped in v0.2

- **`no_grad()` / `enable_grad()`** inference mode — context managers and
  decorators that skip graph construction entirely
- **LR schedulers** — StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
  LinearWarmup, LambdaLR
- **Faster Conv2d** — `_im2col` dual path (`as_strided` for large kernels,
  slice loop for small); see `bench/conv_im2col.py`
- **Model persistence** — `mg.save` / `mg.load` portable `.npz`
- 32 new tests in the 0.2 era (54 total at 0.2 release)

---

## ✅ Shipped in v0.1

- Tensor + dynamic autograd with broadcasting
- Linear, Conv2d (im2col), MaxPool2d, LayerNorm, RMSNorm, Dropout, Embedding, MultiHeadAttention
- Activations: ReLU, GELU, Sigmoid, Tanh, Softmax
- Losses: CrossEntropy (fused), MSE
- Optimisers: SGD (+momentum +Nesterov), AdamW, Lion
- Gradient clipping (norm + value)
- DataLoader + transforms (Normalize, RandomCrop, RandomHorizontalFlip)
- Trainer with gradient accumulation + AMP plumbing
- ONNX export (full graph for the documented op subset)
- Full unit test suite + `gradcheck`

---

## 🛠 Performance & dtype (ongoing)

### CuPy backend
The `microgradx.backend` module already abstracts the array library
behind `xp`. Setting `MICROGRADX_BACKEND=cupy` switches every op to GPU
arrays — remaining hot paths before this is fast:

- ✅ **im2col forward** now uses an `as_strided` view (works for NumPy and
  CuPy, since the kwarg-free `as_strided` call is in the shared `xp` path).
- **col2im** (the conv backward scatter) still loops over the kernel
  dimensions — fine on CPU, worth a `scatter_add` on GPU.
- **Dropout** mask generation should use a GPU-resident PRNG (CuPy's
  `cupy.random` already supports this; just guard the call).

### True mixed precision
Today AMP plumbing is correct but math runs at fp32 (NumPy default).
With CuPy + an fp16 path:
- Layer-level `autocast` dispatches: matmuls → fp16, accumulators → fp32
- Loss scaling already implemented (`GradScaler`) — no API change

Expected speed-up on a typical attention block: 1.6–1.8× on Ampere/Hopper.

### ✅ Memory: gradient checkpointing — shipped
`microgradx.utils.checkpoint(fn, *args)` (also `mg.checkpoint`) re-runs `fn`
during backward instead of saving intermediates.

---

## 🌐 Next — distributed training (Ray)

### Data parallelism
Each worker holds a full model copy and sees a different shard of the
batch. Average gradients across workers before `optimizer.step`.

```python
import ray
from microgradx.dist import DDP

@ray.remote(num_cpus=2)
class Worker:
    def __init__(self, model_factory, opt_factory):
        self.model = model_factory()
        self.opt = opt_factory(self.model.parameters())
    def train_step(self, batch):
        loss = self.model(batch).backward()
        return [p.grad for p in self.model.parameters()]
    def apply_grads(self, avg_grads):
        for p, g in zip(self.model.parameters(), avg_grads):
            p.grad = g
        self.opt.step()
```

Implementation tasks:
- `microgradx.dist.AllReduce` primitive (sum + broadcast via Ray collective)
- `DDP` wrapper module that auto-syncs grads after `backward()`
- Sync-once-per-bucket to overlap reduction with backward (PyTorch DDP trick)
- Throughput target: linear scaling to 16 workers on 1 host with NumPy backend

### Pipeline parallelism (stretch)
Split a model into stages, each on a different worker. Micro-batch
between them so all stages stay busy. Requires:
- A `nn.Stage` wrapper that knows its place in the pipe
- Activation tensors transferred via Ray object store (zero-copy on shared mem)
- 1F1B (one-forward, one-backward) scheduler

---

## 🎚 Quantisation — remaining

### ✅ Dynamic weight-only INT8 — shipped in v0.3
`from microgradx.quant import quantize_dynamic` replaces Linear with
`Int8Linear` (absmax scale, fp32 dequant matmul).

### Still planned
- **Calibration / static activation quant** for true int8 matmul paths
- **Int8Conv1d / Int8Conv2d** variants
- **ONNX export** of the quantised graph (`QuantizeLinear` /
  `DequantizeLinear` nodes)
- **Quantisation-aware training (QAT)** with `FakeQuantize` + STE
- **4-bit weight-only** (group absmax, optional double-quant) for LM inference

---

## 🧰 Extras still worth doing

- **TensorBoard logger**: `mg.utils.TBWriter` writes scalars/histograms
  to a `.tfevents` file (no torch dependency — write protobufs directly)
- **More ops**: `ConvTranspose2d`, bidirectional RNN, `PackSequence`
- **Compile-time graph optimiser**: trace once, fold constants, fuse
  `Add+ReLU` etc. — a 10–20% speedup on small batches

### ✅ Previously listed, now shipped
- Save / load (`.npz`)
- Scheduler module including OneCycleLR + ReduceLROnPlateau
- Conv1d, BatchNorm, RNN/GRU/LSTM
- AvgPool2d / AdaptiveAvgPool2d, GroupNorm, Adam, CSVLogger
