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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        """Kaiming init for Conv layers, constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    model = CIFAR10CNN().to(DEVICE)
    print(f"\n  Model: {type(model).__name__}")
    print(f"  Device: {DEVICE}\n")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # Shape trace
    dummy = torch.randn(2, 3, 32, 32).to(DEVICE)
    with torch.no_grad():
        out = model(dummy)
    print(f"\n  Input  shape: {dummy.shape}")
    print(f"  Output shape: {out.shape}  (logits, no softmax)")


    print("DATA — CIFAR-10 Loaders")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform)
    testset  = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=val_transform)

    import multiprocessing
    N_WORKERS = min(4, multiprocessing.cpu_count())

    train_loader = DataLoader(trainset, batch_size=128, shuffle=True,
                               num_workers=N_WORKERS,
                               pin_memory=torch.cuda.is_available(),
                               drop_last=True)
    test_loader  = DataLoader(testset,  batch_size=256, shuffle=False,
                               num_workers=N_WORKERS,
                               pin_memory=torch.cuda.is_available())

    print(f"\n  Train: {len(trainset)} images → {len(train_loader)} batches of 128")
    print(f"  Test:  {len(testset)} images  → {len(test_loader)} batches of 256")

