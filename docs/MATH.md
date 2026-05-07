# MicroGradX — Mathematical Reference

This document derives the gradient rules every primitive op in the library
implements. If you're adding a new `Function`, read the relevant section
and verify your closed-form against `gradcheck`.

---

## 1. Reverse-mode autodiff in 60 seconds

We compute a scalar loss `L` from input parameters `θ`. We want `∂L/∂θ`.

The chain rule reads, for any intermediate tensor `y = f(x)`:

```
∂L/∂x = ∂L/∂y · ∂y/∂x          (Jacobian-vector product)
```

Reverse-mode never materialises the full Jacobian — for an op with input
shape `m` and output shape `n`, the Jacobian is `n×m`, but `∂L/∂y` is
shape `n` and `∂L/∂x` is shape `m`, so we only ever compute the *product*.

Each `Function` therefore exposes:

* **forward(x₁, …, xₖ)** → `y` (and may stash any tensors needed for backward in `ctx`)
* **backward(g)** → `(∂L/∂x₁, …, ∂L/∂xₖ)` given `g = ∂L/∂y`

`Tensor.backward()` walks the dynamic DAG in reverse topological order
and accumulates these vector-Jacobian products into each input's `.grad`.

---

## 2. Broadcasting

NumPy/PyTorch broadcasting is "implicit replication." If `a` has shape
`(3, 1, 4)` and `b` has shape `(2, 4)`, then `a + b` gets evaluated at the
broadcast shape `(3, 2, 4)`.

For backward, replication ⇒ summation:

> For every axis where the input shape was 1 (or absent) but the output
> shape was > 1, the gradient must be **summed** along that axis.

We centralise this as `_unbroadcast(grad, target_shape)` in `tensor.py`.
Per-op backwards return gradients in the *output* shape and the driver
shrinks them. This makes per-op code dramatically simpler.

---

## 3. Elementwise primitives

| Op            | Forward            | Backward (input grad)                     |
|---------------|--------------------|-------------------------------------------|
| Add           | `a + b`            | `g, g`                                    |
| Sub           | `a - b`            | `g, -g`                                   |
| Mul           | `a · b`            | `g · b, g · a`                            |
| Div           | `a / b`            | `g/b, -g · a / b²`                        |
| Neg           | `-x`               | `-g`                                      |
| Pow (scalar n)| `xⁿ`               | `g · n · xⁿ⁻¹`                            |
| Exp           | `eˣ`               | `g · eˣ` (= `g · y` if y was saved)       |
| Log           | `log x`            | `g / x`                                   |
| Sqrt          | `√x`               | `g / (2√x)`                               |
| Sigmoid       | `σ(x)`             | `g · σ(x)·(1-σ(x))`                       |
| Tanh          | `tanh(x)`          | `g · (1 - tanh(x)²)`                      |
| ReLU          | `max(0, x)`        | `g · 1{x>0}`                              |

### GELU (the `tanh` approximation used by GPT/BERT)

```
y = ½ x · (1 + tanh(c · (x + 0.044715 · x³))),    c = √(2/π)
∂y/∂x = ½(1 + tanh(u)) + ½ · x · (1 - tanh(u)²) · c · (1 + 3 · 0.044715 · x²)
```

---

## 4. Matrix multiply

Forward: `Y = A · B`, with `A: (m, k)`, `B: (k, n)` ⇒ `Y: (m, n)`.

Then for `g = ∂L/∂Y` (shape `(m, n)`):

```
∂L/∂A = g · Bᵀ          shape (m, k)
∂L/∂B = Aᵀ · g          shape (k, n)
```

For batched matmul (`A: (..., m, k)`, `B: (..., k, n)`) we replace `Bᵀ`
with `swapaxes(B, -1, -2)` etc. Done.

---

## 5. Reductions

For `y = sum(x, axis=A)`:

```
∂L/∂x[i] = g[i_reduced]      (gradient broadcasts back to input shape)
```

If `keepdims=False` we re-insert the reduced axes via `expand_dims`
(see `_restore_reduced_axes`), then `broadcast_to(input_shape)`.

For `y = mean(x, axis=A)` divide by `N = prod(x.shape[A])` first.

For `y = max(x, axis=A)` only the *argmax* element gets a non-zero gradient.
Ties are split equally (so the result is continuous w.r.t. inputs).

---

## 6. Softmax & log-softmax (numerically stable)

We compute `z = x - max(x, axis=A, keepdims=True)` so all `exp(z) ∈ [0, 1]`.

### Softmax forward
```
y_i = exp(z_i) / Σⱼ exp(z_j)
```

### Softmax backward — the key identity

```
∂y_i/∂x_j = y_i · (δᵢⱼ - y_j)
```

So if `g = ∂L/∂y`,

```
∂L/∂x_j = Σᵢ gᵢ · y_i · (δᵢⱼ - y_j)
        = y_j · gⱼ - y_j · Σᵢ gᵢ y_i
        = y_j · (gⱼ - s),    where s = Σᵢ gᵢ y_i
```

This is what `Softmax.backward` does in two lines.

### LogSoftmax forward
```
y = z - log(Σⱼ exp(z_j))
```

### LogSoftmax backward

`y_i = log(softmax(x)_i)`, so

```
∂y_i/∂x_j = δᵢⱼ - softmax(x)_j
∂L/∂x_j   = gⱼ - softmax(x)_j · Σᵢ gᵢ
```

---

## 7. Cross-entropy with integer targets

Combined log-softmax + NLL, fused for numerical stability.

```
L = -1/N · Σₙ log_softmax(z_n)[t_n]
```

Closed-form gradient:

```
∂L/∂z_{n,c} = (softmax(z_n)_c - 1{c == t_n}) / N
```

That's literally what `_CrossEntropyFn.backward` writes — no per-element
chain rule needed.

---

## 8. Conv2d (im2col flavour)

Input `X: (N, Cᵢₙ, H, W)`, kernel `W: (Cₒᵤₜ, Cᵢₙ, KH, KW)`, output `Y: (N, Cₒᵤₜ, OH, OW)`.

### Forward

`im2col` rearranges every receptive field into a column:

```
cols : (N · OH · OW, Cᵢₙ · KH · KW)
W₂ₐ  : (Cₒᵤₜ,            Cᵢₙ · KH · KW)
Y_flat = cols · W₂ₐᵀ              shape (N·OH·OW, Cₒᵤₜ)
Y      = reshape & transpose      shape (N, Cₒᵤₜ, OH, OW)
```

### Backward

Let `g = ∂L/∂Y` reshaped to `(N·OH·OW, Cₒᵤₜ)`.

```
∂L/∂W₂ₐ  = gᵀ · cols                     reshape → (Cₒᵤₜ, Cᵢₙ, KH, KW)
∂L/∂cols = g  · W₂ₐ
∂L/∂X    = col2im(∂L/∂cols)             scatter-add overlaps back
∂L/∂b    = sum(g, axes=(0, 2, 3))       per output channel
```

`col2im` is the inverse rearrangement; overlapping patches get *summed*
(not assigned) because the same input pixel contributed to multiple
output positions.

---

## 9. LayerNorm

Forward (along last `D` dims = `axes`):

```
μ  = mean(x, axes, keepdims)
σ² = mean((x - μ)², axes, keepdims)
x̂  = (x - μ) / √(σ² + ε)
y  = γ · x̂ + β
```

Rather than deriving `∂L/∂x` analytically (it's a 4-term mess), we
implement LN as a composition of autograd-aware ops (`mean`, `sub`,
`mul`, `div`, `sqrt`). Backprop handles the algebra.

This is exactly what PyTorch's pure-Python reference impl does. Speed
loss vs the fused kernel is < 2× and we get correctness for free.

---

## 10. Multi-head attention

Per head:

```
Q, K, V = X·W_q, X·W_k, X·W_v          (B, T, d_head)
S = Q · Kᵀ / √d_head                    (B, T, T)
P = softmax(S + mask)                   (B, T, T)
O = P · V                                (B, T, d_head)
```

Multi-head: split `d_model` into `n_heads` groups, run as a single
batched matmul (`B · H` becomes the leading "batch" dim), concat back.

Causal mask is `tril(ones(T, T))` broadcast across batch + head. Where
mask is 0, we add `-1e9` before softmax, which makes attention weight
≈ 0 for those positions.

Backward is just the composition of matmul / softmax / scale-add
backwards — no special code.

---

## 11. Dropout (inverted)

Train:
```
mask ~ Bernoulli(1 - p) / (1 - p)
y = x · mask
```

The `1/(1-p)` scaling means `E[y] = x` regardless of `p`, so we don't
need to do anything at inference time.

Backward: `g · mask`. (Exactly the same mask that was used in forward —
that's what `save_for_backward` is for.)

Eval: identity.

---

## 12. Optimisers

### SGD with Nesterov momentum

```
v ← μ·v + g + wd·θ
θ ← θ - lr · (g + μ·v)            # Nesterov: lookahead by μ·v
```

### AdamW (decoupled weight decay)

```
m ← β₁·m + (1-β₁)·g                 # 1st moment
v ← β₂·v + (1-β₂)·g²                # 2nd moment
m̂ = m/(1-β₁ᵗ);    v̂ = v/(1-β₂ᵗ)     # bias correction
θ ← θ - lr · (m̂/(√v̂ + ε) + wd·θ)    # NB: wd outside the m/√v ratio
```

### Lion

```
c = β₁·m + (1-β₁)·g
θ ← θ - lr · (sign(c) + wd·θ)
m ← β₂·m + (1-β₂)·g
```

Half the state of Adam, robust to lr choice, often faster wall-clock per
optimiser step.

### Gradient clipping

```
clip = max_norm / (‖g‖_global + ε)
if clip < 1: g ← clip · g           (rescale every parameter's grad)
```

---

## 13. Validating new ops with `gradcheck`

```python
from microgradx import gradcheck, Tensor

def my_op(x): ...

x = Tensor(np.random.randn(3, 4), requires_grad=True)
ok = gradcheck(my_op, [x], rtol=1e-5, atol=1e-6)
assert ok
```

The check casts inputs to fp64 internally, runs central differences with
`eps=1e-6`, and compares to your closed-form. If `ok` is `True` your
backward is right (to numerical precision).
