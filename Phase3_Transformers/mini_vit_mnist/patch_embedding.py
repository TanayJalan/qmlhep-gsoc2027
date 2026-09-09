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
import os
import multiprocessing


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches and linearly embeds each.

    Args:
        img_size   : height (= width) of input image
        patch_size : height (= width) of each patch
        in_channels: number of image channels (1 for MNIST, 3 for CIFAR)
        d_model    : output embedding dimension

    Input:  (batch, C, H, W)
    Output: (batch, n_patches, d_model)
    """

    def __init__(self, img_size: int = 28, patch_size: int = 4,
                 in_channels: int = 1, d_model: int = 64):
        super().__init__()
        assert img_size % patch_size == 0, \
            "Image size must be divisible by patch size"

        self.img_size   = img_size
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2
        self.patch_dim  = in_channels * patch_size * patch_size

        # Conv2d with kernel=patch_size and stride=patch_size
        # extracts non-overlapping patches AND linearly projects them.
        # Equivalent to: unfold → reshape → linear, but faster.
        self.projection = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, C, H, W)
        → (batch, n_patches, d_model)
        """
        x = self.projection(x)            # (batch, d_model, H/p, W/p)
        x = x.flatten(2)                  # (batch, d_model, n_patches)
        x = x.transpose(1, 2)            # (batch, n_patches, d_model)
        return x


import sys, atexit
def _clean_exit():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
atexit.register(_clean_exit)


# Test
if __name__ == "__main__":
    patch_emb = PatchEmbedding(img_size=28, patch_size=4,
                                 in_channels=1, d_model=64)
    x_img = torch.randn(8, 1, 28, 28)
    x_pat = patch_emb(x_img)

    print(f"\n  Input  (images): {x_img.shape}")
    print(f"  Output (patches): {x_pat.shape}")
    print(f"  n_patches: {patch_emb.n_patches}  (7×7 grid)")
    print(f"  patch_dim: {patch_emb.patch_dim}  (4×4×1 flattened)")
    print(f"  Projection parameters: "
          f"{sum(p.numel() for p in patch_emb.parameters()):,}")

