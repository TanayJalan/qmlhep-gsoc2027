import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import multiprocessing
import time
import json
import os

try:
    from model import MiniViT
except ImportError:
    from qml_gsoc_tasks.task_VIII_vit_qvit.classical_vit.model import MiniViT

torch.manual_seed(42)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

SAVE_PATH = "vit_mnist_best.pt"
DATA_DIR  = os.path.join(os.path.expanduser("~"), "data")

print(f"  Device: {DEVICE}")
print(f"  Checkpoint: {SAVE_PATH}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),   # MNIST mean/std
])

trainset = torchvision.datasets.MNIST(
    root=DATA_DIR, train=True,  download=True, transform=transform)
testset  = torchvision.datasets.MNIST(
    root=DATA_DIR, train=False, download=True, transform=transform)

N_WORKERS    = 0
train_loader = DataLoader(trainset, batch_size=256, shuffle=True,
                           num_workers=N_WORKERS,
                           pin_memory=torch.cuda.is_available(),
                           drop_last=True)
test_loader  = DataLoader(testset,  batch_size=512, shuffle=False,
                           num_workers=N_WORKERS,
                           pin_memory=torch.cuda.is_available())

print(f"\n  Train: {len(trainset):,}  Test: {len(testset):,}")
print(f"  Train batches: {len(train_loader)}  "
      f"Test batches: {len(test_loader)}")


# ── Model ─────────────────────────────────────────────────────
model = MiniViT(
    img_size=28, patch_size=4, in_channels=1,
    d_model=64, n_heads=8, n_layers=6, d_ff=256,
    n_classes=10, dropout=0.1,
).to(DEVICE)

param_counts = model.count_parameters()
print(f"\n  MiniViT parameters: {param_counts['total']:,}")
print(f"    patch_embed: {param_counts['patch_embed']:,}")
print(f"    pos_embed:   {param_counts['pos_embed']:,}")
print(f"    transformer: {param_counts['transformer']:,}")
print(f"    head:        {param_counts['head']:,}")


# ── Training helpers ─────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for X, y in loader:
        X, y    = X.to(device), y.to(device)
        logits  = model(X)
        correct += (logits.argmax(1) == y).sum().item()
        total   += len(y)
    return correct / total * 100


# ── Optimiser & schedule ──────────────────────────────────────
N_EPOCHS  = 60
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

print(f"\n  Optimizer: AdamW  lr=3e-4  weight_decay=0.05")
print(f"  Scheduler: CosineAnnealingLR  T_max={N_EPOCHS}")
print(f"  Loss:      CrossEntropyLoss  label_smoothing=0.1")
print(f"  Epochs:    {N_EPOCHS}\n")

print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
      f"{'Val Acc':>8}  {'LR':>8}  {'Time':>6}")
print(f"  {'-'*58}")


# ── Training loop ─────────────────────────────────────────────
best_val_acc = 0.0
history      = []

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE)
    val_acc = evaluate(model, test_loader, DEVICE)
    scheduler.step()

    lr_now  = scheduler.get_last_lr()[0]
    elapsed = time.time() - t0

    history.append({
        'epoch':      epoch,
        'train_loss': round(train_loss, 6),
        'train_acc':  round(train_acc, 4),
        'val_acc':    round(val_acc, 4),
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch':      epoch,
            'model_state': model.state_dict(),
            'val_acc':    val_acc,
            'arch': {
                'img_size': 28, 'patch_size': 4, 'in_channels': 1,
                'd_model': 64, 'n_heads': 8, 'n_layers': 6,
                'd_ff': 256, 'n_classes': 10,
            }
        }, SAVE_PATH)
        marker = " ← best"
    else:
        marker = ""

    print(f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2f}%  "
          f"{val_acc:>7.2f}%  {lr_now:>8.6f}  "
          f"{elapsed:>5.1f}s{marker}", flush=True)


# ── Final results ─────────────────────────────────────────────
print(f"\n  Best val accuracy: {best_val_acc:.2f}%")

milestone = "✓  TASK VIII (Part 1) COMPLETE" \
            if best_val_acc >= 98.0 else \
            "⚠  Below 98% — try more epochs or larger d_model"
print(f"  {milestone}")


# ── Per-class accuracy ────────────────────────────────────────
print(f"\n  Per-class accuracy on test set:")
model.load_state_dict(
    torch.load(SAVE_PATH, map_location=DEVICE)['model_state'])
model.eval()

class_correct = [0] * 10
class_total   = [0] * 10

with torch.no_grad():
    for X, y in test_loader:
        X, y   = X.to(DEVICE), y.to(DEVICE)
        preds  = model(X).argmax(1)
        for cls in range(10):
            mask = (y == cls)
            class_correct[cls] += (preds[mask] == y[mask]).sum().item()
            class_total[cls]   += mask.sum().item()

print(f"\n  {'Digit':>6}  {'Correct':>8}  {'Total':>6}  {'Acc':>7}")
print(f"  {'-'*34}")
for i in range(10):
    acc = class_correct[i] / class_total[i] * 100
    print(f"  {i:>6}  {class_correct[i]:>8}  "
          f"{class_total[i]:>6}  {acc:>6.1f}%")


# ── Save results ──────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
results = {
    'best_val_acc': round(best_val_acc, 4),
    'n_params':     param_counts['total'],
    'n_epochs':     N_EPOCHS,
    'architecture': {
        'img_size': 28, 'patch_size': 4, 'n_patches': 49,
        'd_model': 64, 'n_heads': 8, 'n_layers': 6, 'd_ff': 256,
    },
    'per_class_acc': {
        str(i): round(class_correct[i]/class_total[i]*100, 2)
        for i in range(10)
    },
    'history': history,
}

with open("results/vit_mnist_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved: results/vit_mnist_results.json")
print(f"  Checkpoint: {SAVE_PATH}")