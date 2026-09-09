import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import time
import multiprocessing
import sys
import atexit
from tqdm import tqdm

from vit_model import MiniViT

def _clean_exit():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
atexit.register(_clean_exit)

torch.manual_seed(42)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

DATA_DIR = os.path.join(os.path.expanduser("~"), "data")
SAVE_PATH = "vit_mnist_best.pt"


# ── Training helpers ─────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch=1, total_epochs=20):
    model.train()
    total_loss = correct = total = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:>2}/{total_epochs}", leave=False)
    for X, y in pbar:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_sz = len(y)
        loss_val = loss.item()
        total_loss += loss_val * batch_sz
        correct    += (logits.argmax(1) == y).sum().item()
        total      += batch_sz

        pbar.set_postfix({
            'loss': f"{loss_val:.4f}",
            'acc': f"{100.0 * correct / total:.2f}%"
        })

    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for X, y in tqdm(loader, desc="  Evaluating", leave=False):
        X, y    = X.to(device), y.to(device)
        logits  = model(X)
        correct += (logits.argmax(1) == y).sum().item()
        total   += len(y)
    return correct / total * 100



def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))   # MNIST mean / std
    ])

    trainset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=True,  download=True, transform=transform)
    testset  = torchvision.datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform)

    N_WORKERS    = min(4, multiprocessing.cpu_count())
    train_loader = DataLoader(trainset, batch_size=256, shuffle=True,
                               num_workers=N_WORKERS,
                               pin_memory=torch.cuda.is_available(),
                               drop_last=True)
    test_loader  = DataLoader(testset,  batch_size=512, shuffle=False,
                               num_workers=N_WORKERS,
                               pin_memory=torch.cuda.is_available())

    print(f"\n  Train: {len(trainset):,} images → {len(train_loader)} batches")
    print(f"  Test:  {len(testset):,} images  → {len(test_loader)} batches")
    print(f"  Device: {DEVICE}")

    # Instantiate MiniViT
    vit = MiniViT(
        img_size=28, patch_size=4, in_channels=1,
        d_model=64, n_heads=8, n_layers=6, d_ff=256,
        n_classes=10, dropout=0.1
    ).to(DEVICE)

    # ── Training loop ────────────────────────────────────────
    N_EPOCHS  = 30
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(vit.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    print(f"\n  Epochs: {N_EPOCHS}  |  optimizer: AdamW lr=3e-4  "
          f"|  scheduler: CosineAnnealing\n")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Acc':>8}  {'LR':>8}  {'Time':>6}")
    print(f"  {'-'*55}")

    best_val_acc = 0.0

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            vit, train_loader, criterion, optimizer, DEVICE,
            epoch=epoch, total_epochs=N_EPOCHS
        )
        val_acc = evaluate(vit, test_loader, DEVICE)

        scheduler.step()
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':       epoch,
                'model_state': vit.state_dict(),
                'val_acc':     val_acc,
            }, SAVE_PATH)
            marker = " ← best"
        else:
            marker = ""

        print(f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2f}%  "
              f"{val_acc:>7.2f}%  {lr_now:>8.6f}  {elapsed:>5.1f}s{marker}")

    print(f"\n  Best val accuracy: {best_val_acc:.2f}%")

    milestone = " PHASE 3 MILESTONE + TASK VIII (part 1)" \
                if best_val_acc >= 98.0 else \
                " Try more epochs or increase d_model"
    print(f"  {milestone}")


if __name__ == '__main__':
    main()

