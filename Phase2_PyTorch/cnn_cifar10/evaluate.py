import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import time

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

def main():
    print("EVALUATE — Final Results")

    if not os.path.exists(SAVE_PATH):
        print(f"Checkpoint {SAVE_PATH} not found. Please train the model first.")
        return

    from model import CIFAR10CNN
    from train import evaluate
    
    # 1. Setup model
    model = CIFAR10CNN().to(DEVICE)
    
    # 2. Setup dataset and loader
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    
    # 3. Setup criterion
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Load best checkpoint
    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    print(f"\n  Loaded best model from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.2f}%)")

    # Overall accuracy
    _, final_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n  Final test accuracy: {final_acc:.2f}%")

    milestone = "PHASE 2 MILESTONE ACHIEVED" if final_acc >= 80 else \
                "Below 80% — try more epochs or check your transforms"
    print(f"  {milestone}")

    # Per-class accuracy
    print(f"\n  Per-class accuracy:")
    class_correct = [0] * 10
    class_total   = [0] * 10

    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(imgs)
            preds  = logits.argmax(dim=1)
            for cls in range(10):
                mask = labels == cls
                class_correct[cls] += (preds[mask] == labels[mask]).sum().item()
                class_total[cls]   += mask.sum().item()

    print(f"\n  {'Class':<12} {'Correct':>8} {'Total':>6} {'Acc':>7}")
    print(f"  {'-'*36}")
    for i, cls_name in enumerate(CIFAR10_CLASSES):
        acc = class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else 0
        bar = "█" * int(acc // 5)
        print(f"  {cls_name:<12} {class_correct[i]:>8} "
              f"{class_total[i]:>6} {acc:>6.1f}%  {bar}")

    # Training summary
    if 'history' in checkpoint:
        history = checkpoint['history']
        best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
        print(f"\n  Training summary:")
        print(f"    Best val acc:   {max(history['val_acc']):.2f}%  (epoch {best_epoch})")
        print(f"    Final train acc:{history['train_acc'][-1]:.2f}%")
        print(f"    Gap (overfit?): {history['train_acc'][-1] - history['val_acc'][-1]:.2f}%")
    else:
        print("\n  Training summary: Not available in checkpoint. Re-train to save history.")

if __name__ == '__main__':
    main()
