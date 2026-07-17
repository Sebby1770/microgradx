# Changelog

All notable changes to MicroGradX are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-07-17

### Added
- **TransformerEncoder** — stack of N `TransformerEncoderLayer` with optional
  final `LayerNorm`; construct via `TransformerEncoder(layer, num_layers)` or
  `TransformerEncoder.from_config(d_model, nhead, num_layers, …)`.
- **PositionalEncoding** — fixed sinusoidal PE (Vaswani et al.) + dropout;
  supports `batch_first` `(B,T,D)` and time-first `(T,B,D)`.
- **HuberLoss / SmoothL1Loss** — delta/beta threshold, reductions
  `mean` / `sum` / `none`; functional `huber_loss` / `smooth_l1_loss`.
- **CosineAnnealingWarmRestarts** — SGDR-style cosine cycles with `T_0`,
  `T_mult`, `eta_min`.
- **RAdam** — Rectified Adam optimiser (`optim.RAdam`).
- **freeze / unfreeze** — `Module.freeze()`, `Module.unfreeze()`, and
  `Module.requires_grad_(flag)` set `requires_grad` on all parameters.
- **Top-k accuracy** — `microgradx.metrics.accuracy(logits, targets, topk=…)`
  (also `mg.accuracy`); pure NumPy on `.data`.
- **Bilinear upsample** — `Upsample` / `interpolate` now support
  `mode="bilinear"` with integer `scale_factor`.
- **Example**: `examples/encoder_stack_demo.py` — TransformerEncoder + PE +
  RAdam + warm restarts on synthetic data.
- Tests: encoder stack, PE, Huber, warm restarts, RAdam, freeze, metrics,
  bilinear upsample.

### Changed
- Package version bumped to **0.6.0**.
- README / ROADMAP / `pyproject.toml` updated for the new surface area.

## [0.5.0] - 2026-07-17

### Added
- **BCEWithLogitsLoss / BCELoss** — stable binary cross-entropy
  (`max(x,0) - x·y + log(1+exp(-|x|))`) plus probability-space BCE;
  functional helpers `binary_cross_entropy_with_logits` / `binary_cross_entropy`.
- **Label-smoothing CrossEntropy** — optional `label_smoothing` on
  `cross_entropy` / `CrossEntropyLoss` (soft targets `(1-α)·one_hot + α/C`).
- **Upsample / interpolate** — nearest-neighbor upsample for NCHW 4-D tensors
  (`Upsample(scale_factor=2, mode="nearest")`).
- **TransformerEncoderLayer** — MultiHeadAttention + Linear-GELU-Linear FFN
  with LayerNorm, residual connections, and Dropout
  (`d_model`, `nhead`, `dim_feedforward=2048`, `dropout=0.1`).
- **EarlyStopping** — `training.EarlyStopping(patience, mode)` /
  `mg.EarlyStopping`; `step(metric)` returns True when patience is exhausted.
- **EMA** — exponential moving average of parameters with
  `ema.update()` and `with ema.average_parameters(): …` restore context.
- **`manual_seed(seed)`** — seeds Python `random` + NumPy (and CuPy if active).
- **Example**: `examples/transformer_block_demo.py` — one encoder layer,
  label-smoothed CE, EMA, EarlyStopping on synthetic data.
- Tests: BCE, label smoothing, upsample, transformer, early stopping, EMA, seed.

### Changed
- Package version bumped to **0.5.0**.
- README / ROADMAP / `pyproject.toml` updated for the new surface area.

## [0.4.0] - 2026-07-16

### Added
- **Activations**: `LeakyReLU(negative_slope=0.01)`, `SiLU` (Swish), `Softplus`
  (stable `logaddexp` forward, sigmoid backward).
- **Pooling**: `AvgPool2d(kernel_size, stride=None, padding=0)` and
  `AdaptiveAvgPool2d(output_size)` (int or `(H, W)`; `1` = global average pool).
- **GroupNorm** (`num_groups`, `num_channels`, `eps=1e-5`) — batch-size-independent
  channel-group normalisation.
- **Dropout2d** — inverted channel dropout for `(N, C, H, W)` feature maps.
- **Adam** optimiser — classic Kingma & Ba Adam with L2 weight decay folded
  into the gradient (distinct from decoupled AdamW).
- **ReduceLROnPlateau** — reduce LR when a monitored metric plateaus
  (`mode`, `factor`, `patience`, `threshold`, `min_lr`, `cooldown`).
- **CSVLogger** — `from microgradx import CSVLogger`; append named scalars to
  a CSV file (`log(**metrics)` / `close()` / context manager).
- **`Module.apply(fn)`** — recursively apply a function to submodules
  (PyTorch-style, children first).
- **Example**: `examples/cnn_synth.py` — tiny CNN with Conv2d, GroupNorm,
  SiLU, AdaptiveAvgPool2d, Adam, and ReduceLROnPlateau on synthetic
  `(N, 3, 32, 32)` data.
- Tests: activations, pools, GroupNorm, Adam, ReduceLROnPlateau, CSVLogger.

### Changed
- Package version bumped to **0.4.0**.
- README / ROADMAP / `pyproject.toml` updated for the new surface area.

## [0.3.0] - 2026-07-16

### Added
- **Recurrent layers** (`microgradx.nn`): `RNN`, `GRU`, `LSTM` with
  multi-layer support, `batch_first`, optional bias, inter-layer dropout,
  and optional initial hidden (and cell) state. Returns match the
  PyTorch-ish API: `(output, h_n)` / `(output, (h_n, c_n))`.
- **Conv1d** (`microgradx.nn`): 1-D convolution via im2col → GEMM, shapes
  `(N, C, L) → (N, C_out, L_out)`, with stride / padding / bias.
- **OneCycleLR** (`microgradx.optim`): 1cycle policy with warmup to
  `max_lr` then cosine or linear anneal to a tiny final rate
  (`pct_start`, `div_factor`, `final_div_factor`).
- **Dynamic INT8 quantisation** (`microgradx.quant`):
  `quantize_dynamic(model)` replaces `Linear` with `Int8Linear` (absmax
  weight scale, int8 storage, fp32 dequant matmul). Includes `Observer`.
- **Graph utilities**: `count_parameters(model)` and `summary(model,
  input_shape=…)` in `microgradx.utils` (also re-exported at top level).
- **Example**: `examples/seq_classify.py` — tiny GRU sequence classifier
  on synthetic data, with OneCycleLR and optional INT8 eval.
- MIT `LICENSE`.
- Tests for RNN/GRU/LSTM, Conv1d, OneCycleLR, and quantisation.

### Changed
- Package version bumped to **0.3.0**.
- `pyproject.toml` description, classifiers, keywords, and project URLs
  expanded.
- README and roadmap updated for the new surface area.

## [Unreleased] → folded into 0.3.0

Prior unreleased work that shipped with the checkpoint/BatchNorm branch
and is included in this release lineage:

### Added
- **Module buffers**: `register_buffer` / `named_buffers`, and `state_dict` /
  `load_state_dict` now round-trip non-learnable state. This makes
  **BatchNorm running statistics persist through `mg.save` / `mg.load`**.
- **Gradient checkpointing**: `mg.checkpoint(fn, *args)` runs a sub-network
  without storing its activations and recomputes them during backward.
- **BatchNorm1d / BatchNorm2d** (`microgradx.nn`): per-channel normalization
  over the batch (and spatial dims) with running mean/variance and correct
  train vs. eval behavior.

## [0.2.0] - 2026-06-19

### Added
- **Inference mode**: `mg.no_grad()` and `mg.enable_grad()` context managers
  (also usable as decorators), plus `mg.is_grad_enabled()` /
  `mg.set_grad_enabled()`. Inside a `no_grad` region, ops skip graph
  construction entirely — faster evaluation and no retained forward graph.
- **Learning-rate schedulers** (`microgradx.optim`): `StepLR`, `MultiStepLR`,
  `ExponentialLR`, `CosineAnnealingLR`, `LinearWarmup`, `LambdaLR`. Each
  computes its rate from the captured base LR, so they never drift.
- **Model persistence**: `mg.save(model, path)` / `mg.load(path)` to a
  portable, pickle-free `.npz` (`allow_pickle=False`), plus convenience
  `Module.save` / `Module.load`.
- `bench/conv_im2col.py` to reproduce the Conv2d im2col benchmarks.
- 21 new tests (54 total): `no_grad` semantics, every scheduler, save/load
  round-trips and validation, and Conv2d forward-vs-naive equivalence.

### Changed
- **Faster Conv2d**: `_im2col` now dispatches between an `as_strided` view
  (kernels larger than 3×3) and the original slice loop (3×3 and smaller).
  The view removes the per-kernel-position Python loop, giving ~1.6–1.7×
  faster forward passes at 5×5–11×11 while staying par at 3×3. Both paths
  produce byte-identical output.

### Fixed
- `Module.load_state_dict` was a stub containing dead code and an inline
  import; it now validates the key set (`strict=` flag) and every tensor's
  shape before copying.

## [0.1.0] - 2026-05-07

### Added
- Initial release: `Tensor` + dynamic autograd, `nn` layers (Linear, Conv2d,
  MaxPool2d, LayerNorm, RMSNorm, Dropout, Embedding, MultiHeadAttention),
  activations, losses, optimisers (SGD, AdamW, Lion), gradient clipping,
  `DataLoader` + transforms, `Trainer` with grad accumulation + AMP plumbing,
  ONNX export, and a full unit-test suite with `gradcheck`.

[0.4.0]: https://github.com/Sebby1770/microgradx/releases/tag/v0.4.0
[0.3.0]: https://github.com/Sebby1770/microgradx/releases/tag/v0.3.0
[0.2.0]: https://github.com/Sebby1770/microgradx/releases/tag/v0.2.0
[0.1.0]: https://github.com/Sebby1770/microgradx/releases/tag/v0.1.0
