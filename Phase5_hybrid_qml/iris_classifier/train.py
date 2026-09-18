# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pennylane as qml
import numpy as np
import time, json, os

torch.manual_seed(42)
np.random.seed(42)

# %%
iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler()
X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
X_te_t  = torch.tensor(X_te,  dtype=torch.float32)
y_tr_t  = torch.tensor(y_tr,  dtype=torch.long)
y_te_t  = torch.tensor(y_te,  dtype=torch.long)
loader  = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=16, shuffle=True)

print("Data")
print(f"\n  Train={len(X_tr)}  Test={len(X_te)}  Classes={list(iris.target_names)}")


# %%

class ClassicalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16,8), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(8,3))
    def forward(self, x): return self.net(x)

# ── Hybrid Model ──────────────────────────────────────────
N_QUBITS, N_LAYERS = 4, 3
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RX(weights[l,i,0], wires=i)
            qml.RY(weights[l,i,1], wires=i)
            qml.RZ(weights[l,i,2], wires=i)
        for i in range(N_QUBITS-1):
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, N_QUBITS), nn.Tanh())
        self.q_layer = qml.qnn.TorchLayer(circuit, {"weights": (N_LAYERS, N_QUBITS, 3)})
        self.head    = nn.Linear(N_QUBITS, 3)
    def forward(self, x):
        x = self.encoder(x).double()
        x = self.q_layer(x)
        return self.head(x.float())

# ── Generic train loop ────────────────────────────────────
def train(model, loader, Xte, yte, n_epochs, lr, name):
    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=lr)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    n_p  = sum(p.numel() for p in model.parameters())
    best_acc = 0.0

    print(f"\n  {name}  params={n_p}")
    print(f"  {'Ep':>4}  {'Loss':>8}  {'Tr%':>7}  {'Val%':>7}  {'t':>5}")
    print(f"  {'-'*36}")

    for ep in range(1, n_epochs+1):
        t0 = time.time()
        model.train()
        tl = tc = tt = 0
        for xb, yb in loader:
            opt.zero_grad()
            out  = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step()
            tl += loss.item()*len(yb); tc += (out.argmax(1)==yb).sum().item(); tt += len(yb)
        sch.step()
        tr_acc = tc/tt*100; tr_loss = tl/tt
        model.eval()
        with torch.no_grad():
            va = (model(Xte).argmax(1)==yte).float().mean()*100
        if va.item() > best_acc: best_acc = va.item()
        if ep % 30 == 0 or ep == 1:
            print(f"  {ep:>4}  {tr_loss:>8.4f}  {tr_acc:>6.1f}%  {va.item():>6.1f}%  {time.time()-t0:>4.1f}s")

    model.eval()
    with torch.no_grad():
        fa = (model(Xte).argmax(1)==yte).float().mean()*100
    print(f"\n  Final test accuracy: {fa.item():.2f}%  |  Best val: {best_acc:.2f}%")
    return fa.item(), best_acc

# ── Train both ────────────────────────────────────────────
print("Training Classical MLP")
cls_model = ClassicalMLP()
cls_acc, _ = train(cls_model, loader, X_te_t, y_te_t, 150, 1e-3, "ClassicalMLP")

print("Training Hybrid Model")
hyb_model = HybridModel()
hyb_acc, _ = train(hyb_model, loader, X_te_t, y_te_t, 150, 5e-3, "HybridModel")

# ── Comparison ────────────────────────────────────────────
print("Results Comparison")
cls_p = sum(p.numel() for p in cls_model.parameters())
hyb_p = sum(p.numel() for p in hyb_model.parameters())
hyb_q = sum(p.numel() for n,p in hyb_model.named_parameters() if 'weights' in n)

print(f"""
  Model            Params   Test Acc
  ─────────────────────────────────
  ClassicalMLP     {cls_p:>6}   {cls_acc:.2f}%
  HybridModel      {hyb_p:>6}   {hyb_acc:.2f}%
    classical       {hyb_p-hyb_q:>5}
    quantum         {hyb_q:>5}

  Parameter reduction: {(1-hyb_p/cls_p)*100:.1f}%
  Accuracy delta:      {hyb_acc-cls_acc:+.2f}%
""")

# Per-class breakdown
print("  Per-class accuracy (hybrid):")
hyb_model.eval()
with torch.no_grad():
    preds = hyb_model(X_te_t).argmax(1)
for i, name in enumerate(iris.target_names):
    mask = y_te_t == i
    acc  = (preds[mask] == y_te_t[mask]).float().mean()*100
    print(f"    {name:<12}: {acc.item():.1f}%")

# Save
os.makedirs("results", exist_ok=True)
torch.save({'model_state': hyb_model.state_dict(), 'acc': hyb_acc}, "results/hybrid_best.pt")
print(f"\n  Saved: results/hybrid_best.pt")

milestone = hyb_acc >= 90.0

# %%