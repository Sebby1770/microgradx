"""
ONNX exporter via tracing.

Approach (matches torch.onnx.export's eager-trace path):
  1. Run a forward pass with a representative dummy input
  2. Walk the dynamic graph in reverse topological order from the output
  3. Map each MicroGradX Function to its ONNX equivalent
  4. Emit a `ModelProto` (if `onnx` is installed) or a JSON dump otherwise

The op map covers the common ML primitives — extend OP_MAP for new ops.
Anything in the graph that isn't in OP_MAP raises NotImplementedError so
you don't silently ship broken models.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Callable
import json
import numpy as np

from microgradx.tensor import Tensor


# ---- op-name → ONNX op-type map ----
# Each entry: (onnx_op, attr_extractor)
# attr_extractor gets the Context and returns dict of attributes.
def _attrs_softmax(ctx):
    return {"axis": ctx.axis if hasattr(ctx, "axis") else -1}

def _attrs_reshape(ctx):
    return {"_shape_input": list(ctx.input_shape) if False else None,
            "_target_shape": None}  # handled specially via initializer

def _attrs_reduce(ctx):
    keepdims = int(ctx.keepdims) if hasattr(ctx, "keepdims") else 0
    axes = ctx.axis if hasattr(ctx, "axis") else None
    if axes is None:
        return {"keepdims": keepdims}
    if isinstance(axes, int):
        axes = [axes]
    return {"axes": list(axes), "keepdims": keepdims}

def _attrs_transpose(ctx):
    return {"perm": list(ctx.axes)}

def _attrs_pow(ctx):
    return {"_pow_exponent": float(ctx.exponent)}

def _attrs_conv(ctx):
    sh, sw = ctx.stride
    ph, pw = ctx.padding
    return {"strides": [sh, sw], "pads": [ph, pw, ph, pw]}


OP_MAP: Dict[str, Tuple[str, Callable]] = {
    "Add":         ("Add",        lambda ctx: {}),
    "Sub":         ("Sub",        lambda ctx: {}),
    "Mul":         ("Mul",        lambda ctx: {}),
    "Div":         ("Div",        lambda ctx: {}),
    "Neg":         ("Neg",        lambda ctx: {}),
    "Pow":         ("Pow",        _attrs_pow),
    "MatMul":      ("MatMul",     lambda ctx: {}),
    "ReLU":        ("Relu",       lambda ctx: {}),
    "Sigmoid":     ("Sigmoid",    lambda ctx: {}),
    "Tanh":        ("Tanh",       lambda ctx: {}),
    "GELU":        ("Gelu",       lambda ctx: {}),
    "Exp":         ("Exp",        lambda ctx: {}),
    "Log":         ("Log",        lambda ctx: {}),
    "Sqrt":        ("Sqrt",       lambda ctx: {}),
    "Softmax":     ("Softmax",    _attrs_softmax),
    "LogSoftmax":  ("LogSoftmax", _attrs_softmax),
    "Sum":         ("ReduceSum",  _attrs_reduce),
    "Mean":        ("ReduceMean", _attrs_reduce),
    "Max":         ("ReduceMax",  _attrs_reduce),
    "Min":         ("ReduceMin",  _attrs_reduce),
    "Reshape":     ("Reshape",    lambda ctx: {}),
    "Transpose":   ("Transpose",  _attrs_transpose),
    "_Conv2dFn":   ("Conv",       _attrs_conv),
    "_DropoutFn":  ("Dropout",    lambda ctx: {"is_training_mode": 0}),
}


def _trace(output: Tensor) -> List[Tensor]:
    """Iterative reverse-topo from the output."""
    topo, visited = [], set()
    stack = [(output, False)]
    while stack:
        node, done = stack.pop()
        nid = id(node)
        if done:
            topo.append(node)
            continue
        if nid in visited:
            continue
        visited.add(nid)
        stack.append((node, True))
        if node._ctx is not None:
            for inp in node._ctx.input_tensors:
                if inp is not None and id(inp) not in visited:
                    stack.append((inp, False))
    return topo


def export_to_dict(
    model_forward: Callable[[Tensor], Tensor],
    dummy_input: Tensor,
    parameters: Optional[List[Tensor]] = None,
) -> dict:
    """Trace a forward pass and produce a JSON-serialisable graph spec.

    If `onnx` is installed, you can pass this dict to `to_onnx_proto`.
    """
    output = model_forward(dummy_input)
    topo = _trace(output)

    name_of: Dict[int, str] = {}
    next_id = [0]
    def fresh(prefix="t"):
        s = f"{prefix}_{next_id[0]}"
        next_id[0] += 1
        return s

    inputs: List[dict] = []
    outputs: List[dict] = []
    initializers: List[dict] = []
    nodes: List[dict] = []

    name_of[id(dummy_input)] = "input"
    inputs.append({"name": "input", "shape": list(dummy_input.shape),
                   "dtype": str(dummy_input.dtype)})
    if parameters:
        for i, p in enumerate(parameters):
            n = f"param_{i}"
            name_of[id(p)] = n
            initializers.append({"name": n, "shape": list(p.shape),
                                  "dtype": str(p.dtype),
                                  "data": p.numpy().flatten().tolist()})

    for t in topo:
        if t._ctx is None:
            if id(t) not in name_of:
                # Constant input we didn't declare — register as initializer
                cname = fresh("const")
                name_of[id(t)] = cname
                initializers.append({"name": cname, "shape": list(t.shape),
                                      "dtype": str(t.dtype),
                                      "data": t.numpy().flatten().tolist()})
            continue
        ctx = t._ctx
        op_name = ctx.fn.__name__
        if op_name not in OP_MAP:
            raise NotImplementedError(
                f"ONNX export: no mapping for op '{op_name}'. "
                f"Add it to OP_MAP."
            )
        onnx_op, attr_fn = OP_MAP[op_name]
        attrs = attr_fn(ctx)

        # Resolve input names; create initializers for unknown leaves
        in_names = []
        for inp in ctx.input_tensors:
            if inp is None:
                continue
            if id(inp) not in name_of:
                cname = fresh("const")
                name_of[id(inp)] = cname
                initializers.append({"name": cname, "shape": list(inp.shape),
                                      "dtype": str(inp.dtype),
                                      "data": inp.numpy().flatten().tolist()})
            in_names.append(name_of[id(inp)])

        # Special-case Reshape: ONNX needs a "shape" tensor as 2nd input
        if op_name == "Reshape":
            shape_name = fresh("shape")
            initializers.append({"name": shape_name, "shape": [len(t.shape)],
                                  "dtype": "int64", "data": list(t.shape)})
            in_names.append(shape_name)

        # Pow exponent goes through as a tensor input in ONNX
        if op_name == "Pow":
            exp_name = fresh("pow_exp")
            initializers.append({"name": exp_name, "shape": [],
                                  "dtype": "float32",
                                  "data": [attrs.pop("_pow_exponent")]})
            in_names.append(exp_name)

        out_name = fresh("t")
        name_of[id(t)] = out_name
        nodes.append({
            "op": onnx_op,
            "inputs": in_names,
            "outputs": [out_name],
            "attrs": attrs,
        })

    # Mark final
    out_name = name_of[id(output)]
    outputs.append({"name": out_name, "shape": list(output.shape),
                    "dtype": str(output.dtype)})

    return {
        "ir_version": 7,
        "producer_name": "microgradx",
        "graph": {
            "name": "main",
            "inputs": inputs,
            "outputs": outputs,
            "initializers": initializers,
            "nodes": nodes,
        },
    }


def export(
    model_forward: Callable[[Tensor], Tensor],
    dummy_input: Tensor,
    path: str,
    parameters: Optional[List[Tensor]] = None,
):
    """Export to `path`. If `path.endswith('.onnx')` and `onnx` is importable,
    write a real ONNX ModelProto. Otherwise write JSON (still inspectable)."""
    spec = export_to_dict(model_forward, dummy_input, parameters)
    if path.endswith(".onnx"):
        try:
            import onnx
            from onnx import helper, TensorProto
            model_proto = _spec_to_onnx_proto(spec, onnx, helper, TensorProto)
            onnx.save(model_proto, path)
            return path
        except ImportError:
            print("[microgradx] `onnx` not installed — falling back to JSON.")
            path = path[:-5] + ".json"
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)
    return path


_DTYPE_TO_ONNX = {
    "float32": "FLOAT",
    "float16": "FLOAT16",
    "float64": "DOUBLE",
    "int64":   "INT64",
    "int32":   "INT32",
    "bool":    "BOOL",
}


def _spec_to_onnx_proto(spec, onnx_mod, helper, TensorProto):
    g = spec["graph"]

    def _t(dtype):
        return getattr(TensorProto, _DTYPE_TO_ONNX.get(str(dtype), "FLOAT"))

    inits = []
    for init in g["initializers"]:
        arr = np.asarray(init["data"]).reshape(init["shape"]).astype(
            np.dtype(init["dtype"])
        )
        inits.append(helper.make_tensor(
            name=init["name"], data_type=_t(init["dtype"]),
            dims=init["shape"], vals=arr.flatten().tolist(),
        ))

    nodes = []
    for n in g["nodes"]:
        nodes.append(helper.make_node(
            op_type=n["op"], inputs=n["inputs"], outputs=n["outputs"],
            **n["attrs"],
        ))

    inputs = [helper.make_tensor_value_info(i["name"], _t(i["dtype"]), i["shape"])
              for i in g["inputs"]]
    outputs = [helper.make_tensor_value_info(o["name"], _t(o["dtype"]), o["shape"])
               for o in g["outputs"]]

    graph = helper.make_graph(nodes, g["name"], inputs, outputs, initializer=inits)
    return helper.make_model(graph, producer_name=spec["producer_name"])
