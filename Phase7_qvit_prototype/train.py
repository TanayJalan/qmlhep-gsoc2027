import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pennylane as qml
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import math
import time
import multiprocessing

torch.manual_seed(42)

try:
    from qvit_model import MiniQViT
except ImportError:
    from Phase7_qvit_prototype.qvit_model import MiniQViT

# PennyLane default.qubit requires float64 precision; PyTorch MPS backend does not support float64
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"  Device: {DEVICE}")
print(f"  PennyLane: {qml.__version__}")

print("train.py — Training QVIT on MNIST")

DATA_DIR   = os.path.join(os.path.expanduser("~"), "data")
SAVE_PATH  = "qvit_mnist_best.pt"

transform  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(
    root=DATA_DIR, train=True,  download=True, transform=transform)
testset  = torchvision.datasets.MNIST(
    root=DATA_DIR, train=False, download=True, transform=transform)

# Use a subset for speed (full MNIST × quantum circuit = slow on CPU)
N_TRAIN   = 4000    # 400 images per digit for solid pattern learning
N_TEST    = 1000    # 1000 validation images

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        correct += (model(X).argmax(1) == y).sum().item()
        total   += len(y)
    return correct / total * 100


if __name__ == "__main__":
    train_sub = Subset(trainset, range(N_TRAIN))
    test_sub  = Subset(testset,  range(N_TEST))

    train_loader = DataLoader(train_sub, batch_size=32, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_sub,  batch_size=64, shuffle=False, num_workers=0)

    print(f"\n  Train subset: {N_TRAIN} images  |  Test subset: {N_TEST} images")
    print(f"  (Use full MNIST with GPU for publication-quality results)")

    qvit = MiniQViT(n_heads=2, n_blocks=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(qvit.parameters(), lr=7e-4, weight_decay=1e-4)
    N_EPOCHS  = 30     # 30 epochs for deep convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    print(f"\n  Epochs: {N_EPOCHS}  (in-memory CPU training)")
    print(f"  For full training: increase N_TRAIN to 5000+ and use MPS/CUDA\n")
    print(f"  {'Epoch':>5}  {'Loss':>9}  {'Tr Acc':>8}  {'Val Acc':>9}  {'Time':>6}")
    print(f"  {'-'*46}")

    best_val = 0.0

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            qvit, train_loader, criterion, optimizer, DEVICE)
        val_acc = evaluate(qvit, test_loader, DEVICE)
        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            torch.save({'epoch': epoch, 'state': qvit.state_dict(),
                        'val_acc': val_acc}, SAVE_PATH)
            marker = " <- best"
        else:
            marker = ""

        print(f"  {epoch:>5}  {tr_loss:>9.4f}  {tr_acc:>7.2f}%  "
              f"{val_acc:>8.2f}%  {time.time()-t0:>5.1f}s{marker}")

    print(f"\n  Best val accuracy: {best_val:.2f}%")

