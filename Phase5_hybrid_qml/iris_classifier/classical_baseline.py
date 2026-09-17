# %% 0

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


torch.manual_seed(42)
np.random.seed(42)

# %% 

iris = load_iris()
X, y = iris.data, iris.target

n_features, n_classes = X.shape[1], len(np.unique(y))

print(f"\n  Samples:    {len(X)}")
print(f"  Features:   {n_features}  ({', '.join(iris.feature_names)})")
print(f"  Classes:    {n_classes}  ({', '.join(iris.target_names)})")
print(f"  Class dist: {np.bincount(y).tolist()}  (50 each)")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardise features (zero mean, unit variance)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_ds     = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")
print(f"Features standardised: mean≈0, std≈1")

# %%

class ClassicalMLP(nn.Module):
    def __init__(self, in_features=4, n_classes=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)


model   = ClassicalMLP()
n_params = sum(p.numel() for p in model.parameters())
print(f"\nClassicalMLP parameters: {n_params}")
for name, param in model.named_parameters():
    print(f"{name:<25}{tuple(param.shape)}")

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
N_EPOCHS  = 150

print(f"\nOptimizer: Adam  lr=1e-3")
print(f"Epochs:{N_EPOCHS}")
print(f"\n{'Epoch':>6}{'Train Loss':>10}{'Train Acc':>10}{'Val Acc':>8}")
print(f"{'-'*40}")

history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    total_loss = correct = total = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        out  = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(yb)
        correct+= (out.argmax(1) == yb).sum().item()
        total+= len(yb)

    scheduler.step()

    train_loss = total_loss / total
    train_acc = correct / total * 100

    model.eval()
    with torch.no_grad():
        val_out = model(X_test_t)
        val_acc = (val_out.argmax(1) == y_test_t).float().mean() * 100

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc.item())

    if epoch % 30 == 0 or epoch == 1:
        print(f"{epoch:>6}  {train_loss:>10.4f}  "
              f"{train_acc:>9.2f}%  {val_acc.item():>7.2f}%")

# Final evaluation
model.eval()
with torch.no_grad():
    final_out = model(X_test_t)
    final_acc = (final_out.argmax(1) == y_test_t).float().mean() * 100
    final_preds = final_out.argmax(1)

print(f"\nFinal test accuracy: {final_acc.item():.2f}%")
print(f"Best val accuracy: {max(history['val_acc']):.2f}%")

 # %%
for cls_idx, cls_name in enumerate(iris.target_names):
    mask = y_test_t == cls_idx
    cls_acc = (final_preds[mask] == y_test_t[mask]).float().mean() * 100
    print(f"  {cls_name:<12}: {cls_acc.item():.1f}%")

# %%

import json, os

results = {
    "model":      "ClassicalMLP",
    "params":     n_params,
    "final_acc":  round(final_acc.item(), 4),
    "best_acc":   round(max(history['val_acc']), 4),
    "n_epochs":   N_EPOCHS,
    "history":    history,
}
os.makedirs("results", exist_ok=True)
with open("results/classical_baseline.json", "w") as f:
    json.dump(results, f, indent=2)

# %%