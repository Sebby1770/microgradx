# Changelog

All notable changes to MicroGradX are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Module buffers**: `register_buffer` / `named_buffers`, and `state_dict` /
  `load_state_dict` now round-trip non-learnable state. This makes
  **BatchNorm running statistics persist through `mg.save` / `mg.load`** — a
  loaded BatchNorm model now evaluates identically to the original (previously
  the running mean/var were silently dropped).
- **Gradient checkpointing**: `mg.checkpoint(fn, *args)` runs a sub-network
  without storing its activations and recomputes them during backward —
  trading compute for memory on deep blocks. Numerically transparent:
  identical forward output, input gradients, and parameter gradients vs. the
  direct call.
- **BatchNorm1d / BatchNorm2d** (`microgradx.nn`): per-channel normalization
  over the batch (and spatial dims) with running mean/variance and correct
  train vs. eval behavior, so a single eval sample is handled properly.
- 11 new tests (65 total): checkpoint equivalence and BatchNorm
  (normalization, running-stat updates, eval path, single-sample,
  `gradcheck`, affine gradients).

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

[0.2.0]: https://github.com/Sebby1770/microgradx/releases/tag/v0.2.0
[0.1.0]: https://github.com/Sebby1770/microgradx/releases/tag/v0.1.0
