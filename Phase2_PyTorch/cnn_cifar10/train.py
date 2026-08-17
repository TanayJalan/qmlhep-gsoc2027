import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time

from model import CIFAR10CNN

torch.manual_seed(42)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

DATA_DIR   = os.path.join(os.path.expanduser("~"), "data")
SAVE_PATH  = "cnn_cifar10_best.pt"

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking = True)
        y_batch = y_batch.to(device, non_blocking = True)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)
    
    return total_loss / total, correct / total * 100

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on a dataset. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / total, correct / total * 100

def main():
    print('Train')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    # 2. Instantiate the model
    model = CIFAR10CNN().to(device)
    # 3. Setup optimizer and loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    N_EPOCHS = 50

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    print(f"\n  Optimizer:  AdamW  lr=1e-3  weight_decay=5e-4")
    print(f"  Scheduler:  CosineAnnealingLR  T_max={N_EPOCHS}")
    print(f"  Loss:       CrossEntropyLoss  label_smoothing=0.1")
    print(f"  Epochs:     {N_EPOCHS}")
    print(f"\n  ({'GPU/MPS training — should take ~5-10 min' if DEVICE.type != 'cpu' else 'CPU training — consider pushing to Colab for full run'})")
    print(f"\n  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>9}  {'Val Acc':>8}  {'LR':>8}  {'Time':>6}")
    print(f"  {'-'*65}")

    best_val_acc  = 0.0
    history       = {'train_loss': [], 'train_acc': [],
                     'val_loss':   [], 'val_acc':   []}

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        val_loss,   val_acc   = evaluate(
            model, test_loader, criterion, DEVICE)

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':      epoch,
                'model_state': model.state_dict(),
                'opt_state':   optimizer.state_dict(),
                'val_acc':     val_acc,
                'history':     history,
            }, SAVE_PATH)
            marker = " ← best"
        else:
            marker = ""

        print(f"  {epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2f}%  "
              f"{val_loss:>8.4f}  {val_acc:>7.2f}%  {lr_now:>8.6f}  "
              f"{elapsed:>5.1f}s{marker}")

if __name__ == '__main__':
    main()
