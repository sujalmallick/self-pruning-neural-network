# Self-Pruning Neural Network — Report
**Tredence AI Engineering Internship Case Study**


## 1. Why does an L1 Penalty on Sigmoid Gates encourage Sparsity?

### The Core Intuition

The sigmoid function maps any real number to the open interval **(0, 1)**:

```
gate = σ(gate_score) = 1 / (1 + e^(−gate_score))
```

During optimization, we learn **soft gates** for the L1 penalty and use **hard 0/1 gates in the forward pass** with a straight-through estimator (STE):

```
soft_gate = sigmoid(gate_score * temperature)
hard_gate = 1 if soft_gate >= 0.5 else 0
effective_weight = weight * hard_gate
```

This gives exact zeros during training while still allowing gradients to flow through `soft_gate`.

### Why L1 and not L2?

The **L1 norm** (sum of absolute values) adds a constant gradient to each gate regardless of its current magnitude:

```
d/d(gate) [ |gate| ] = sign(gate) = +1   (since gate > 0 after sigmoid)
```

This means every active gate gets a constant "push toward zero" during gradient descent. Gates that are **not useful for accuracy** will have no countervailing gradient from the classification loss, so they will drift all the way to zero.

In contrast, **L2** (sum of squares) gives a gradient proportional to `2 * gate`. This shrinks large gates quickly but barely moves small gates, leading to many small-but-nonzero values — not true sparsity.

### The Loss Formulation

```
Total Loss = CrossEntropyLoss(logits, labels) + λ × Σ(all gates)
```

- **CrossEntropyLoss** pulls gate values up (to preserve accuracy).
- **λ × Σ(gates)** pushes gate values down toward zero.
- The tension between the two forces creates a natural selection: gates for **important weights** survive; gates for **redundant weights** collapse to zero.
- A higher **λ** increases sparsity pressure, potentially sacrificing some accuracy.

---

## 2. Results Table

The current script runs three lambda values:

```
LAMBDAS = [1e-7, 1e-6, 1e-5]
```

| Lambda (λ) | Test Accuracy | Sparsity Level (%) |
|:----------:|:-------------:|:------------------:|
| 1e-07 (Low) | 57.03% | 94.84% |
| 1e-06 (Medium) | 54.91% | 98.33% |
| 1e-05 (High) | 48.93% | 99.69% |

> **Note:** Exact values vary per run due to random initialisation, hardware, and epoch count. Use the printed console table (`Lambda | Test Accuracy | Sparsity Level`) from each run as the authoritative result for your environment.

### Interpretation

- **Low λ (1e-7):** Gentle sparsity pressure — 94.8% pruned, accuracy stays at 57%
- **Medium λ (1e-6):** Moderate pruning — 98.3% pruned, slight accuracy drop to 54.9%
- **High λ (1e-5):** Aggressive pruning — 99.7% pruned, accuracy drops to 48.9%

---

## 3. Gate Value Distribution Plot

The plot (`gate_distributions.png`) shows the histogram of **hard gate values** after training for each λ.

### What a successful result looks like:

```
Count
 |
 ████  ← Large spike at 0 (pruned weights)
 |████
 |████
 |████
 |████
 |████                    ██ ← Cluster near 1 (active weights)
 +─────────────────────────────── Gate Value
 0                               1
```


- A **large spike at 0** means many weights are exactly pruned by hard gates.
- A **secondary cluster near 1** means surviving weights stay active.
- As λ increases, the spike at 0 grows taller and the cluster near 1 shrinks.

The red dashed line marks the hard-gate threshold (0.5) used for pruning decisions.
As λ increases from 1e-7 to 1e-5, the pruned bar grows and the active bar nearly disappears.

---

## 4. Design Decisions Explained

### PrunableLinear — Gradient Flow

The custom layer registers `gate_scores` as `nn.Parameter`, which means PyTorch's autograd engine automatically tracks gradients through:

```
soft_gate   = sigmoid(gate_score * T)    # differentiable
hard_gate   = step(soft_gate - 0.5)      # 0 or 1 in forward
pruned_w    = weight * hard_gate         # exact pruning in forward
output      = F.linear(x, pruned_w, b)  # standard linear op
```

Using STE, backpropagation uses the gradient path of `soft_gate` while forward uses `hard_gate`. This preserves trainability and still yields exact zeros in active weights.

### Why BatchNorm + Dropout?

CIFAR-10 with a pure feed-forward network tends to overfit. BatchNorm stabilises training; Dropout provides an independent regularisation signal so the L1 penalty is not competing with dropout alone.

### Hard-Gate Threshold Choice (0.5)

The model applies a hard gate with threshold `0.5` on the sigmoid output during forward pass:

- `soft_gate >= 0.5` → active connection (`hard_gate = 1`)
- `soft_gate < 0.5`  → pruned connection (`hard_gate = 0`)

Because forward uses hard gates, sparsity is measured directly as the percentage of exact zeros.

---

## 5. How to Run

```bash
# Install dependencies
pip install torch torchvision matplotlib

# Run the experiment (downloads CIFAR-10 automatically)
python self_pruning_network.py
```

Output files:
- Console: epoch-by-epoch logs + final results table
- `gate_distributions.png`: gate histogram for all three λ values
