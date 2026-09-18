
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math
import time
import os

from data_loader import get_jet_dataloaders, N_FEATURES, N_FEATURES_REAL, CLASS_NAMES

torch.manual_seed(42)
np.random.seed(42)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"  Device: {DEVICE}")


class JetTransformerBlock(nn.Module):
    """Pre-LN Transformer encoder block for jet sequences."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class JetTransformer(nn.Module):
    
    def __init__(self,
                 n_features:   int = N_FEATURES,
                 d_model:      int = 32,
                 n_heads:      int = 4,
                 n_layers:     int = 3,
                 d_ff:         int = 64,
                 n_classes:    int = 2,
                 dropout:      float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.pos_drop  = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            JetTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm      = nn.LayerNorm(d_model)
        self.head      = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, n_classes),
        )

    def forward(self, x):
        """x: (batch, n_const, n_features)"""
        # Embed constituents
        x = self.embedding(x)         # (B, T, d_model)
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1)            

        return self.head(x)           



print("Data")

# ── Toggle real vs synthetic data ─────────────────────────────────────────────
# Real Zenodo dataset (train.h5, val.h5, test.h5) — set DATA_DIR to use it.
# Input features for real data: E, PX, PY, PZ, pT  (5 features, col index 0-4)
# Input features for synthetic:  14 features (N_FEATURES from data_loader)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')   # Phase6_hep_papers/data/
USE_REAL  = os.path.isdir(DATA_DIR) and os.path.isfile(os.path.join(DATA_DIR, 'train.h5'))

if USE_REAL:
    INPUT_FEATURES = 5          # E, PX, PY, PZ, pT
    N_TRAIN, N_VAL, N_TEST = 50_000, 5_000, 5_000
else:
    INPUT_FEATURES = N_FEATURES  # 14 synthetic features
    N_TRAIN, N_VAL, N_TEST = 3000, 500, 500

TOP_K = 30   # top-30 constituents by pT

train_loader, val_loader, test_loader, sample_shape = get_jet_dataloaders(
    n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
    mode='sequence', top_k=TOP_K, batch_size=256,
    data_dir=DATA_DIR if USE_REAL else None,
)

print(f"\n  Dataset: {'REAL (Zenodo)' if USE_REAL else 'SYNTHETIC'}")
print(f"  Train: {N_TRAIN}  Val: {N_VAL}  Test: {N_TEST}")
print(f"  Input shape per jet: {sample_shape}  (top-{TOP_K} constituents)")
print(f"  Classes: {CLASS_NAMES}")


print("Training JetTransformer")

model = JetTransformer(
    n_features=INPUT_FEATURES, d_model=64, n_heads=4,
    n_layers=4, d_ff=128, n_classes=2, dropout=0.1
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"\n  Model parameters: {n_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
N_EPOCHS  = 50 if USE_REAL else 30
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)


def _roc_auc(y_true, y_score):
    """Compute ROC-AUC using sklearn if available, else trapezoidal approximation."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        sorted_idx = np.argsort(-y_score)
        y_true_s   = y_true[sorted_idx]
        n_pos = y_true.sum()
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        tp = np.cumsum(y_true_s)
        fp = np.cumsum(1 - y_true_s)
        tpr = tp / n_pos
        fpr = fp / n_neg
        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        auc = trapz_fn(tpr, fpr)
        return float(abs(auc))


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb  = xb.to(device), yb.to(device)
            logits  = model(xb)
            probs   = torch.softmax(logits, dim=1)[:, 1]
            correct += (logits.argmax(1) == yb).sum().item()
            total   += len(yb)
            all_probs.append(probs.cpu())
            all_labels.append(yb.cpu())
    acc = correct / total * 100
    # AUC via ROC curve
    probs_all  = torch.cat(all_probs).numpy()
    labels_all = torch.cat(all_labels).numpy()
    auc = _roc_auc(labels_all, probs_all)
    return acc, auc


print(f"\n  {'Epoch':>5}  {'Loss':>9}  {'Tr Acc':>8}  "
      f"{'Val Acc':>8}  {'Val AUC':>9}  {'Time':>5}")
print(f"  {'-'*50}")

best_auc  = 0.0
best_acc  = 0.0
history   = []

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    model.train()
    total_loss = correct = total = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(yb)
        correct    += (model(xb).argmax(1) == yb).sum().item()
        total      += len(yb)

    scheduler.step()
    tr_loss = total_loss / total
    tr_acc  = correct / total * 100
    val_acc, val_auc = evaluate(model, val_loader, DEVICE)

    if val_auc > best_auc:
        best_auc = val_auc
        best_acc = val_acc
        torch.save(model.state_dict(), 'jet_transformer_best.pt')

    history.append({'loss': tr_loss, 'tr_acc': tr_acc,
                    'val_acc': val_acc, 'auc': val_auc})

    if epoch % 5 == 0 or epoch == 1:
        print(f"  {epoch:>5}  {tr_loss:>9.4f}  {tr_acc:>7.2f}%  "
              f"{val_acc:>7.2f}%  {val_auc:>9.4f}  "
              f"{time.time()-t0:>4.1f}s")


model.load_state_dict(torch.load('jet_transformer_best.pt',
                                   map_location=DEVICE))
test_acc, test_auc = evaluate(model, test_loader, DEVICE)

print("Phase 6 Milestone — Results")
print(f"""
  JetTransformer on Top Quark Tagging (synthetic data):

    Test Accuracy:  {test_acc:.2f}%
    Test AUC:       {test_auc:.4f}
    Best Val AUC:   {best_auc:.4f}
    Parameters:     {n_params:,}
""")

milestone = test_acc >= 75.0 and test_auc >= 0.75
print(milestone)

import json, os
os.makedirs('results', exist_ok=True)
with open('results/classical_transformer.json', 'w') as f:
    json.dump({
        'test_acc': round(test_acc, 4),
        'test_auc': round(test_auc, 4),
        'best_val_auc': round(best_auc, 4),
        'n_params': n_params,
        'history': history,
    }, f, indent=2)
