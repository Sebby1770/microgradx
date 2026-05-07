# MicroGradX — Roadmap

What's working today vs what's planned. Status: **v0.1.0**.

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

## 🛠 v0.2 — performance & dtype

### CuPy backend
The `microgradx.backend` module already abstracts the array library
behind `xp`. Setting `MICROGRADX_BACKEND=cupy` switches every op to GPU
arrays — but a few hot paths need attention before this is fast:

- **im2col / col2im** currently uses Python loops over the kernel
  dimensions. On GPU this is too slow; replace with stride-trick views
  (`as_strided`) for both NumPy and CuPy.
- **Dropout** mask generation should use a GPU-resident PRNG (CuPy's
  `cupy.random` already supports this; just guard the call).

### True mixed precision
Today AMP plumbing is correct but math runs at fp32 (NumPy default).
With CuPy + an fp16 path:
- Layer-level `autocast` dispatches: matmuls → fp16, accumulators → fp32
- Loss scaling already implemented (`GradScaler`) — no API change

Expected speed-up on a typical attention block: 1.6–1.8× on Ampere/Hopper.

### Memory: gradient checkpointing
Add `microgradx.utils.checkpoint(fn, *args)` that re-runs `fn` during
backward instead of saving intermediates. Crucial for transformers with
long sequences.

---

## 🌐 v0.3 — distributed training (Ray)

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

The driver does:
```python
grads_per_worker = ray.get([w.train_step.remote(b) for w, b in zip(workers, batches)])
avg = [sum(gs) / len(gs) for gs in zip(*grads_per_worker)]
ray.get([w.apply_grads.remote(avg) for w in workers])
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

## 🎚 v0.4 — quantisation

### Post-training INT8 quantisation
For inference-only deployment.

```python
from microgradx.quant import quantize_dynamic
qmodel = quantize_dynamic(model, dtype="int8", layers=(nn.Linear, nn.Conv2d))
```

Steps:
1. **Calibration pass**: forward a few representative batches; track
   per-tensor `min/max` (or per-channel for weights) → derive `scale, zero_point`.
2. **Replace** every targeted layer with an `Int8Linear` / `Int8Conv2d`
   variant whose forward is:
   ```
   x_q = round(x / scale_x) + zp_x         (uint8)
   y_q = matmul_int8(x_q, w_q)             (int32 accumulator)
   y   = (y_q - zp_y_offset) · scale_x · scale_w
   ```
3. **Bias** stays in fp32 and is added after dequant (small numerical win).

Implementation tasks:
- `Quantizer` (per-tensor or per-channel), `Observer` (records min/max during calibration)
- `Int8Linear`, `Int8Conv2d` modules
- ONNX export of the quantised graph (uses `QuantizeLinear` / `DequantizeLinear` nodes)

### Quantisation-aware training (QAT)
Insert "fake-quant" nodes during training:
```
y = (round(x / scale) - zp) · scale       # forward
∂L/∂x = ∂L/∂y · 1[x_min ≤ x ≤ x_max]      # straight-through estimator
```

Add `microgradx.nn.FakeQuantize` and a `prepare_qat(model)` helper that
wraps Linear/Conv2d in-place. Train as normal; the model learns to
tolerate the quantisation noise.

### 4-bit weight-only (LLM era)
For inference of decoder-only LMs:
- Per-block (group=64) absmax scaling: store weight in int4 + fp16 scale
- Custom `int4_linear` op that dequantises on the fly during matmul
- Optional double-quant of the scale tensor (QLoRA recipe)

---

## 🧰 v0.5 — extras worth doing

- **Save / load**: `mg.save(model, "x.npz")` / `mg.load("x.npz")` using
  numpy's `.npz` archive over the state_dict
- **TensorBoard logger**: `mg.utils.TBWriter` writes scalars/histograms
  to a `.tfevents` file (no torch dependency — write protobufs directly)
- **Scheduler module**: `optim.lr_scheduler.{StepLR, CosineAnnealingLR, OneCycleLR}`
- **More ops**: `Conv1d`, `ConvTranspose2d`, `BatchNorm{1,2}d`, `RNN/GRU/LSTM`
- **Compile-time graph optimiser**: trace the graph once, fold constants,
  fuse `Add+ReLU` etc. — a 10–20% speedup on small batches
