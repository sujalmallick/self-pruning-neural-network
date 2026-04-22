# Self-Pruning Neural Network Report

This project trains a CIFAR-10 classifier that learns to prune its own connections while training.

## 1. Why L1 on sigmoid gates creates sparsity

Each weight has a learnable gate score, and the gate value is:

```
gate = sigmoid(gate_score)
```

During training, the total loss is:

```
Total Loss = CrossEntropyLoss + lambda * sum(all soft gates)
```

Cross-entropy tries to keep useful connections active, while the L1 term adds constant pressure to shrink gate values. If a connection is not important for accuracy, this pressure keeps pushing its soft gate down. In this implementation, the forward pass uses a hard decision (`soft_gate >= 0.5`) with a straight-through estimator, so pruned connections become exactly zero in the effective weights.

## 2. Experiment setup

Current script config:

```
EPOCHS = 50
LAMBDAS = [1e-7, 1e-6, 1e-5]
```

## 3. Results (latest run)

| Lambda (lambda) | Test Accuracy | Sparsity Level (%) |
|:---------------:|:-------------:|:------------------:|
| 1e-07 (Low)     | 57.03%        | 94.84%             |
| 1e-06 (Medium)  | 54.91%        | 98.33%             |
| 1e-05 (High)    | 48.93%        | 99.69%             |

Interpretation:

- Low lambda (1e-7): pruning starts, but the model still keeps enough active paths to hold accuracy near 57%.
- Medium lambda (1e-6): stronger pruning pressure increases sparsity and causes a modest accuracy drop.
- High lambda (1e-5): pruning becomes very aggressive, leaving very few active weights and a clear accuracy decline.

## 4. Gate distribution plot

The saved plot ([gate_distributions.png](gate_distributions.png)) uses hard gate values (0 or 1) as a bar chart:

- Bar at 0: number of pruned weights.
- Bar at 1: number of active weights.

As lambda increases from 1e-7 to 1e-5, the pruned bar grows and the active bar shrinks.

## 5. Design notes

- Soft gates are used for the sparsity penalty (`sum(sigmoid(gate_scores))`) so gradients flow properly.
- Hard gates are used in the forward pass with STE so pruning is real in the model computation, not only in reporting.
- Sparsity is measured from hard gates as the percentage of exact zeros.

## 6. Run instructions

```bash
pip install torch torchvision matplotlib
python self_pruning_network.py
```

Outputs:

- Console logs: per-epoch loss/sparsity and final table.
- [gate_distributions.png](gate_distributions.png): pruning vs active counts for each lambda.
