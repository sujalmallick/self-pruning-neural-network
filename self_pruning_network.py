#  IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#  PART 1: PrunableLinear Layer

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that learns which weights to prune.

    How it works:
    - Each weight w_ij has a corresponding learnable scalar gate_score_ij.
    - gate_score_ij is passed through a Sigmoid to produce gate_ij ∈ (0, 1).
    - The effective weight is: pruned_weight = weight * gate  (element-wise).
    - When gate → 0, the weight is effectively removed (pruned).
    - The L1 penalty on gates pushes them toward 0 during training.

    Gradient flow:
    - d(Loss)/d(weight)     → normal chain rule through linear op
    - d(Loss)/d(gate_score) → flows through sigmoid and element-wise multiply
    Both parameters are registered via nn.Parameter so the optimizer updates them.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.gate_threshold = 0.5

        # Standard weight and bias — same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        self.gate_scores = nn.Parameter(torch.ones(out_features, in_features) * 2.0)

        # Initialise weights with Kaiming uniform (same default as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def _soft_gates(self) -> torch.Tensor:
        """Continuous gates in (0, 1), used by the sparsity regularizer."""
        return torch.sigmoid(self.gate_scores)

    def _hard_gates_ste(self) -> torch.Tensor:
        """Hard forward gates with straight-through gradients."""
        soft = self._soft_gates()
        hard = (soft >= self.gate_threshold).float()
        return hard.detach() - soft.detach() + soft

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Compute hard gates in forward, soft gates in backward.
        gates = self._hard_gates_ste()                   # shape: (out, in)

        # Step 2: Element-wise multiply weights by gates
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Step 3: Standard linear transform  y = x·W^T + b
        # We use F.linear which computes  x @ pruned_weights.T + bias
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return hard 0/1 gate values (detached) for sparsity analysis."""
        return (self._soft_gates() >= self.gate_threshold).float().detach()

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of this layer's gates used in the total loss."""
        gates = self._soft_gates()
        return gates.sum()   # all values positive after sigmoid, so |gate| = gate



#  NETWORK DEFINITION


class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 classification using PrunableLinear layers.

    Architecture:
        Input  : 32×32×3 = 3072 features (flattened)
        Hidden1: 512 neurons  (PrunableLinear + ReLU + BN)
        Hidden2: 256 neurons  (PrunableLinear + ReLU + BN)
        Hidden3: 128 neurons  (PrunableLinear + ReLU + BN)
        Output : 10 classes   (PrunableLinear, no activation — CE loss handles softmax)

    All linear layers are PrunableLinear so every weight has a learnable gate.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = PrunableLinear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = PrunableLinear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = PrunableLinear(128, 10)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)           # Flatten: (B, 3, 32, 32) → (B, 3072)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))

        x = self.fc4(x)                      # logits — no activation
        return x

    def prunable_layers(self):
        """Generator yielding all PrunableLinear layers in the network."""
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def total_sparsity_loss(self) -> torch.Tensor:
        """Sum of L1 norms of all gates across all PrunableLinear layers."""
        return sum(layer.sparsity_loss() for layer in self.prunable_layers())

    def sparsity_level(self) -> float:
        """
        Fraction of weights that are exactly pruned by hard gates (gate == 0).
        """
        all_gates = torch.cat(
            [layer.get_gates().flatten() for layer in self.prunable_layers()]
        )
        pruned = (all_gates == 0).float().sum()
        return (pruned / all_gates.numel()).item() * 100.0   # percentage



#  PART 2 & 3: TRAINING AND EVALUATION


def get_cifar10_loaders(batch_size: int = 128):
    """
    Download CIFAR-10 and return train/test DataLoaders.
    Normalisation stats: mean and std per channel (standard CIFAR-10 values).
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),    # light augmentation
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, lam: float):
    """
    Run one full pass over the training set.

    Loss = CrossEntropyLoss(logits, labels) + λ * Σ||gates||_1

    Returns:
        avg_total_loss   : average total loss over the epoch
        avg_cls_loss     : average classification loss
        avg_sparse_loss  : average sparsity penalty (before λ scaling)
    """
    model.train()
    total_loss_sum = cls_loss_sum = sparse_loss_sum = 0.0
    n_batches = len(loader)

    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        cls_loss    = criterion(logits, labels)
        sparse_loss = model.total_sparsity_loss()
        total_loss  = cls_loss + lam * sparse_loss

        total_loss.backward()
        optimizer.step()

        total_loss_sum  += total_loss.item()
        cls_loss_sum    += cls_loss.item()
        sparse_loss_sum += sparse_loss.item()

    return (total_loss_sum / n_batches,
            cls_loss_sum   / n_batches,
            sparse_loss_sum/ n_batches)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return test accuracy (%) over the full test set."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100.0


def run_experiment(lam: float, epochs: int, train_loader, test_loader, device):
    """
    Train a SelfPruningNet with a given λ and return results dict.
    """
    print(f"\n{'='*60}")
    print(f"  λ = {lam}   |   Epochs = {epochs}")
    print(f"{'='*60}")

    model = SelfPruningNet().to(device)

    # Adam optimizer — updates both weights AND gate_scores
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Cosine annealing LR schedule for stable convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"total_loss": [], "cls_loss": [], "sparse_loss": [], "sparsity_pct": []}

    for epoch in range(1, epochs + 1):
        total_l, cls_l, sparse_l = train_one_epoch(model, train_loader, optimizer, device, lam)
        scheduler.step()

        sparsity = model.sparsity_level()
        history["total_loss"].append(total_l)
        history["cls_loss"].append(cls_l)
        history["sparse_loss"].append(sparse_l)
        history["sparsity_pct"].append(sparsity)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Total={total_l:.4f}  CE={cls_l:.4f}  "
                  f"Sparse={sparse_l:.1f}  | Sparsity={sparsity:.1f}%")

    # Final evaluation
    test_acc  = evaluate(model, test_loader, device)
    sparsity  = model.sparsity_level()

    print(f"\nTest Accuracy : {test_acc:.2f}%")
    print(f"Sparsity Level: {sparsity:.2f}%")

    # Collect gate values from all layers for plotting
    all_gates = torch.cat(
        [layer.get_gates().flatten() for layer in model.prunable_layers()]
    ).cpu().numpy()

    all_soft_gates = torch.cat(
        [layer._soft_gates().detach().flatten() for layer in model.prunable_layers()]
    ).cpu().numpy()

    print("Min gate:", all_gates.min())
    print("Max gate:", all_gates.max())

    return {
        "lam"       : lam,
        "test_acc"  : test_acc,
        "sparsity"  : sparsity,
        "all_gates" : all_gates,
        "all_soft_gates" : all_soft_gates,
        "history"   : history,
        "model"     : model,
    }


def plot_gate_distributions(results: list, save_path: str = "gate_distributions.png"):
    """
    For each λ, plot the distribution of final gate values.
    A well-pruned model shows a large spike at 0 and a smaller cluster near 1.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        gates = res["all_soft_gates"]
        ax.hist(gates, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(
            f"λ = {res['lam']}\n"
            f"Acc = {res['test_acc']:.1f}%  |  Sparsity = {res['sparsity']:.1f}%",
            fontsize=12
        )
        ax.set_xlabel("Soft Gate Value (0 = pruned, 1 = active)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1.2, label="Hard gate threshold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Gate Value Distributions — Self-Pruning Network", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved → {save_path}")


def print_results_table(results: list):
    """Print a formatted results table to console."""
    print("\n" + "="*55)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity Level':>15}")
    print("="*55)
    for r in results:
        print(f"  {r['lam']:<12} {r['test_acc']:>14.2f}%  {r['sparsity']:>13.2f}%")
    print("="*55)



#  MAIN


if __name__ == "__main__":

    # ── Config 
    EPOCHS     = 30
    BATCH_SIZE = 128
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Three λ values: low / medium / high sparsity pressure
    LAMBDAS = [1e-4, 1e-3, 1e-2]

    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS}")

    # Data
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    #Experiments
    results = []
    for lam in LAMBDAS:
        res = run_experiment(lam, EPOCHS, train_loader, test_loader, DEVICE)
        results.append(res)

    # Summary Table 
    print_results_table(results)

    # Plot
    plot_gate_distributions(results, save_path="gate_distributions.png")
