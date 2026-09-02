import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

torch.manual_seed(0)

class SyntheticDataset(Dataset):
    
    def __init__(self, n_samples=500, noise=0.1, seed=42):
        np.random.seed(seed)
        x       = np.linspace(-np.pi, np.pi, n_samples).astype(np.float32)
        y       = np.sin(x) + noise * np.random.randn(n_samples).astype(np.float32)
        self.X  = torch.from_numpy(x).unsqueeze(1)   # (N, 1)
        self.y  = torch.from_numpy(y).unsqueeze(1)   # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = SyntheticDataset()
print(f"\n  Dataset length: {len(dataset)}")

x0, y0 = dataset[0]
print(f"  dataset[0]:  x={x0.item():.4f}  y={y0.item():.4f}")

x5, y5 = dataset[5]
print(f"  dataset[5]:  x={x5.item():.4f}  y={y5.item():.4f}")

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,        # 0 = load in main process (safe on all platforms)
    pin_memory=False,     # set True when using CUDA GPU
    drop_last=False
)

print(f"\n  DataLoader over {len(dataset)} samples, batch_size=32:")
print(f"  Number of batches: {len(loader)}")

# Inspect one batch
X_batch, y_batch = next(iter(loader))
print(f"\n  First batch:")
print(f"    X_batch shape: {X_batch.shape}")
print(f"    y_batch shape: {y_batch.shape}")
print(f"    dtype:         {X_batch.dtype}")

# Iterate over all batches
total_samples = 0
for i, (xb, yb) in enumerate(loader):
    total_samples += len(xb)
print(f"\n  Total samples seen in one epoch: {total_samples}")
print(f"  (matches dataset length: {len(dataset)})")


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),          # crop after padding
    transforms.RandomHorizontalFlip(p=0.5),        # 50% horizontal flip
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2),        # colour augmentation
    transforms.ToTensor(),                          # → (C, H, W) float [0,1]
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),             # CIFAR-10 RGB means
        std=(0.2023, 0.1994, 0.2010)               # CIFAR-10 RGB stds
    )
])

# Validation / test transforms (no random ops)
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

print(f"\n  Train transform pipeline:")
for t in train_transform.transforms:
    print(f"{t}")

print(f"\n  Val transform pipeline:")
for t in val_transform.transforms:
    print(f"{t}")

DATA_DIR = os.path.join(os.path.expanduser("~"), "data")

print(f"\n  Downloading CIFAR-10 to {DATA_DIR} ...")
print(f"  (Already downloaded? torchvision will skip re-download)\n")

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Download train and test sets
trainset = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True, download=True, transform=train_transform
)
testset  = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=val_transform
)

print(f"\n  Train set: {len(trainset)} images")
print(f"  Test  set: {len(testset)} images")
print(f"  Classes:   {CIFAR10_CLASSES}")

# Inspect a single sample
img, label = trainset[0]
print(f"\n  Sample [0]:")
print(f"    image shape: {img.shape}    (C, H, W)")
print(f"    dtype:       {img.dtype}")
print(f"    value range: [{img.min():.3f}, {img.max():.3f}]  (normalised)")
print(f"    label:       {label} = '{CIFAR10_CLASSES[label]}'")

# Class distribution
labels_all = [trainset[i][1] for i in range(len(trainset))]
print(f"\n  Class distribution (train):")
for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
    count = labels_all.count(cls_idx)
    bar   = "-" * (count // 200)
    print(f"    {cls_idx} {cls_name:<12} {count:>5}  {bar}")


print("SECTION 5 — Production DataLoaders")

# Determine num_workers safely
# Note: On macOS with Python 3.8+, multiprocessing uses 'spawn' by default.
# DataLoader with num_workers > 0 requires 'if __name__ == "__main__":' guard,
# otherwise num_workers=0 is safest for standalone interactive scripts.
import multiprocessing
n_cpu     = multiprocessing.cpu_count()
n_workers = 0    # Set to 0 on macOS without if __name__ == '__main__' guard

train_loader = DataLoader(
    trainset,
    batch_size=128,
    shuffle=True,
    num_workers=n_workers,
    pin_memory=torch.cuda.is_available(),   # only pin for CUDA
    drop_last=True
)

test_loader  = DataLoader(
    testset,
    batch_size=256,
    shuffle=False,              # no shuffle for evaluation
    num_workers=n_workers,
    pin_memory=torch.cuda.is_available()
)

print(f"\n  Train loader: {len(train_loader)} batches of 128")
print(f"  Test  loader: {len(test_loader)} batches of 256")
print(f"  num_workers:  {n_workers}")

# Verify a batch
imgs, labels = next(iter(train_loader))
print(f"\n  Train batch:")
print(f"    images shape: {imgs.shape}    (batch, C, H, W)")
print(f"    labels shape: {labels.shape}")
print(f"    unique labels in batch: "
      f"{sorted(labels.unique().tolist())}")

raw_trainset = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True, download=False,
    transform=transforms.ToTensor()    # just convert, no normalise
)
raw_loader = DataLoader(raw_trainset, batch_size=1000,
                         shuffle=False, num_workers=0)

channel_sum    = torch.zeros(3)
channel_sum_sq = torch.zeros(3)
n_pixels       = 0

for imgs, _ in raw_loader:
    # imgs: (batch, 3, 32, 32)
    channel_sum    += imgs.sum(dim=[0, 2, 3])       # sum over batch, H, W
    channel_sum_sq += (imgs ** 2).sum(dim=[0, 2, 3])
    n_pixels       += imgs.shape[0] * 32 * 32       # batch * H * W

mean = channel_sum / n_pixels
std  = torch.sqrt(channel_sum_sq / n_pixels - mean ** 2)

print(f"  Computed mean: {mean.tolist()}")
print(f"  Computed std:  {std.tolist()}")
print(f"  Expected mean: [0.4914, 0.4822, 0.4465]")
print(f"  Expected std:  [0.2023, 0.1994, 0.2010]")
print(f"\n  Match: {torch.allclose(mean, torch.tensor([0.4914, 0.4822, 0.4465]), atol=1e-3)}")
