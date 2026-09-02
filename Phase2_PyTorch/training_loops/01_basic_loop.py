import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(0)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"  Device: {DEVICE}")

class TwoLayerMLP(nn.Module):
    def __init__(self, n_in=2, n_hidden=4, n_out=1):
        super().__init__()          # always call this first
        self.fc1 = nn.Linear(n_in,     n_hidden)   # W1 + b1 inside
        self.fc2 = nn.Linear(n_hidden, n_out)       # W2 + b2 inside

    def forward(self, x):
        x = torch.tanh(self.fc1(x))     # hidden layer + tanh
        x = torch.sigmoid(self.fc2(x))  # output layer + sigmoid
        return x


model = TwoLayerMLP()
print(f"\n  Model:\n{model}")
print(f"\n  Parameters:")
total = 0
for name, param in model.named_parameters():
    print(f"    {name:<15} shape={tuple(param.shape)}  "
          f"numel={param.numel()}")
    total += param.numel()
print(f"  Total: {total} parameters")

# Move to device
model = model.to(DEVICE)
print(f"\n  Model moved to: {next(model.parameters()).device}")

# BCE
pred   = torch.tensor([[0.9], [0.1], [0.8], [0.2]])
target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
bce    = nn.BCELoss()(pred, target)
print(f"  BCELoss:              {bce.item():.4f}")

# CrossEntropy (logits in, no softmax needed)
logits = torch.tensor([[2.0, 1.0, 0.1],
                        [0.5, 2.5, 0.3]])
labels = torch.tensor([0, 1])          # class indices, not one-hot
ce     = nn.CrossEntropyLoss()(logits, labels)
print(f"  CrossEntropyLoss:     {ce.item():.4f}  (logits in, handles softmax)")

# MSE
pred_r  = torch.tensor([1.5, 2.5, 3.5])
target_r = torch.tensor([1.0, 2.0, 3.0])
mse     = nn.MSELoss()(pred_r, target_r)
print(f"  MSELoss:{mse.item():.4f}")

# SGD: same as Phase 1 manual update
model_sgd = TwoLayerMLP().to(DEVICE)
opt_sgd   = optim.SGD(model_sgd.parameters(), lr=0.5)

# Adam: adaptive per-parameter learning rates
model_adam = TwoLayerMLP().to(DEVICE)
opt_adam   = optim.Adam(model_adam.parameters(), lr=1e-3)

# AdamW: Adam + proper weight decay (use this by default)
model_adamw = TwoLayerMLP().to(DEVICE)
opt_adamw   = optim.AdamW(model_adamw.parameters(), lr=1e-3, weight_decay=1e-4)

print(f"  SGD:   {opt_sgd}")
print(f"  Adam:  {opt_adam}")
print(f"  AdamW: {opt_adamw}")

# Data
X_xor = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y_xor = torch.tensor([[0],[1],[1],[0]],          dtype=torch.float32)

# Expand with noise
torch.manual_seed(1)
N      = 200
X_data = X_xor.repeat(N // 4, 1) + 0.1 * torch.randn(N, 2)
y_data = y_xor.repeat(N // 4, 1)
perm   = torch.randperm(N)
X_data, y_data = X_data[perm].to(DEVICE), y_data[perm].to(DEVICE)

# Model, loss, optimiser
net       = TwoLayerMLP(n_in=2, n_hidden=8, n_out=1).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.05)

print(f"\n  Training {type(net).__name__} on XOR ({N} samples)...")
print(f"  {'Epoch':>6}  {'Loss':>10}  {'Acc':>8}")
print(f"  {'-'*28}")

EPOCHS    = 2000
log_every = 400

for epoch in range(1, EPOCHS + 1):
    # ── Training step 
    net.train()                          # set training mode
    optimizer.zero_grad()                # 1. zero gradients
    pred = net(X_data)                   # 2. forward pass
    loss = criterion(pred, y_data)       # 3. compute loss
    loss.backward()                      # 4. backward pass
    optimizer.step()                     # 5. update params


    if epoch % log_every == 0 or epoch == 1:
        net.eval()
        with torch.no_grad():
            pred_eval = net(X_data)
            acc = ((pred_eval > 0.5).float() == y_data).float().mean() * 100
        print(f"  {epoch:>6}  {loss.item():>10.6f}  {acc.item():>7.1f}%")

# Final evaluation on base XOR
net.eval()
with torch.no_grad():
    X_base = X_xor.to(DEVICE)
    y_base = y_xor.to(DEVICE)
    probs  = net(X_base)
    preds  = (probs > 0.5).float()

print(f"\n  XOR truth table:")
for i in range(4):
    correct = "✓" if preds[i].item() == y_base[i].item() else "✗"
    print(f"    {X_xor[i].tolist()} → {probs[i].item():.4f} "
          f"→ {int(preds[i].item())}  {correct}")

acc_final = (preds == y_base).float().mean().item() * 100
print(f"\n  XOR accuracy: {acc_final:.1f}%")

import os, tempfile

# Save
save_path = os.path.join(tempfile.gettempdir(), "xor_model.pt")
torch.save(net.state_dict(), save_path)
print(f"  Saved to: {save_path}")
print(f"  Keys: {list(net.state_dict().keys())}")

# Load into a fresh model
net2 = TwoLayerMLP(n_in=2, n_hidden=8, n_out=1).to(DEVICE)
net2.load_state_dict(torch.load(save_path, map_location=DEVICE))
net2.eval()

with torch.no_grad():
    probs2 = net2(X_xor.to(DEVICE))

print(f"\n  Loaded model predictions: {probs2.squeeze().tolist()}")
print(f"  Match original:           "
      f"{'✓' if torch.allclose(probs, probs2) else '✗'}")