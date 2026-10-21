
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
import json
import os
import time

from graph_construction import generate_jet, build_knn_graph, N_NODE_FEATURES
from model_gcn import JetGCN
from model_gat import JetGAT

torch.manual_seed(42)
np.random.seed(42)


# ── Hyperparameters ───────────────────────────────────────────
N_TRAIN    = 1500
N_VAL      = 300
N_TEST     = 300
K_NN       = 8
BATCH_SIZE = 32
N_EPOCHS   = 40
LR         = 3e-4


# ── Dataset ───────────────────────────────────────────────────
def make_dataset(n: int, k: int = K_NN, seed: int = 42):
    """Build list of PyG Data objects from synthetic jets."""
    from torch_geometric.data import Data
    rng    = np.random.RandomState(seed)
    graphs = []
    for _ in range(n):
        is_top = rng.rand() > 0.5
        feats  = generate_jet(n_constituents=20, is_top=is_top, rng=rng)
        graph  = build_knn_graph(feats, k=k)
        graph.y = torch.tensor([int(is_top)], dtype=torch.long)
        graphs.append(graph)
    return graphs


print("Building datasets...")
train_graphs = make_dataset(N_TRAIN, seed=42)
val_graphs   = make_dataset(N_VAL,   seed=43)
test_graphs  = make_dataset(N_TEST,  seed=44)

train_loader = PyGDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = PyGDataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = PyGDataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {N_TRAIN}  Val: {N_VAL}  Test: {N_TEST}")
print(f"k-NN graph k={K_NN}  |  batch_size={BATCH_SIZE}\n")


# ── Metrics ───────────────────────────────────────────────────
def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC via trapezoidal rule (no sklearn)."""
    idx  = np.argsort(-y_score)
    yt   = y_true[idx]
    n_p  = int(y_true.sum())
    n_n  = len(y_true) - n_p
    if n_p == 0 or n_n == 0:
        return 0.5
    tpr  = np.cumsum(yt)       / n_p
    fpr  = np.cumsum(1 - yt)   / n_n
    return float(abs(np.trapz(tpr, fpr)))


@torch.no_grad()
def evaluate(model: nn.Module, loader: PyGDataLoader) -> tuple:
    """Return (accuracy %, AUC)."""
    model.eval()
    all_probs, all_labels = [], []
    correct = total = 0

    for batch in loader:
        logits = model(batch)
        probs  = F.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total   += len(batch.y)
        all_probs.append(probs.numpy())
        all_labels.append(batch.y.numpy())

    acc  = correct / total * 100
    auc  = roc_auc(np.concatenate(all_labels), np.concatenate(all_probs))
    return acc, auc


# ── Training loop ─────────────────────────────────────────────
def train_model(model: nn.Module, name: str,
                n_epochs: int = N_EPOCHS,
                lr: float = LR) -> dict:
    """Train one model. Returns results dict."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    n_params = model.count_parameters()
    best_auc = 0.0
    save_path = f"{name}_best.pt"
    history   = []

    print(f"{'='*55}")
    print(f"  Training {name}  |  parameters: {n_params:,}")
    print(f"{'='*55}")
    print(f"  {'Ep':>3}  {'Loss':>8}  {'TrAcc':>7}  "
          f"{'ValAcc':>7}  {'ValAUC':>8}  {'t':>4}")
    print(f"  {'-'*44}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = correct = total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch)
            loss   = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch.y)
            correct    += (logits.argmax(1) == batch.y).sum().item()
            total      += len(batch.y)

        scheduler.step()
        tr_loss = total_loss / total
        tr_acc  = correct / total * 100
        val_acc, val_auc = evaluate(model, val_loader)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)

        history.append({
            'epoch': epoch, 'loss': tr_loss,
            'tr_acc': tr_acc, 'val_acc': val_acc, 'val_auc': val_auc,
        })

        if epoch % 8 == 0 or epoch == 1:
            print(f"  {epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>6.1f}%  "
                  f"{val_acc:>6.1f}%  {val_auc:>8.4f}  "
                  f"{time.time()-t0:>3.1f}s")

    # Load best and evaluate on test set
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    test_acc, test_auc = evaluate(model, test_loader)

    print(f"\n  {name}  →  test acc: {test_acc:.2f}%  "
          f"test AUC: {test_auc:.4f}  "
          f"best val AUC: {best_auc:.4f}\n")

    return {
        'name':        name,
        'params':      n_params,
        'test_acc':    round(test_acc, 4),
        'test_auc':    round(test_auc, 4),
        'best_val_auc': round(best_auc, 4),
        'history':     history,
    }


# ── Run ───────────────────────────────────────────────────────
gcn = JetGCN(in_channels=N_NODE_FEATURES)
gat = JetGAT(in_channels=N_NODE_FEATURES)

res_gcn = train_model(gcn, "JetGCN")
res_gat = train_model(gat, "JetGAT")


# ── Comparison table ──────────────────────────────────────────
print("=" * 58)
print("  Task II — Results")
print("=" * 58)

rows = [res_gcn, res_gat]
print(f"\n  {'Model':<14}  {'Params':>8}  {'Test Acc':>9}  {'Test AUC':>9}")
print(f"  {'-'*46}")
for r in rows:
    print(f"  {r['name']:<14}  {r['params']:>8,}  "
          f"{r['test_acc']:>8.2f}%  {r['test_auc']:>9.4f}")

winner = max(rows, key=lambda r: r['test_auc'])
print(f"\n  Best model: {winner['name']}  (AUC = {winner['test_auc']:.4f})")

print(f"""
  Reference (real dataset):
    ParticleNet (EdgeConv):     AUC = 0.9858
    Classical Transformer:      AUC = 0.9850
    GCN baseline:               AUC ~ 0.970
    GAT baseline:               AUC ~ 0.975

  Note: synthetic data is simpler than real HEP jets.
  Download zenodo.org/record/2603256 for benchmark results.
""")


# ── Save results ──────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
with open("results/task_II_results.json", "w") as f:
    json.dump({'gcn': res_gcn, 'gat': res_gat}, f, indent=2)
print("  Saved: results/task_II_results.json")
