"""Post-training quantisation helpers.

Currently: dynamic (weight-only) INT8 replacement of Linear layers for
faster / smaller inference. Activations stay in float32; weights are
stored as int8 + a per-tensor absmax scale and dequantised on the fly.
"""
from microgradx.quant.dynamic import quantize_dynamic, Int8Linear, Observer

__all__ = ["quantize_dynamic", "Int8Linear", "Observer"]
