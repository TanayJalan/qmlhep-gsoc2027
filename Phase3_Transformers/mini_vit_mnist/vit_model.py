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
import sys
import atexit

from patch_embedding import PatchEmbedding
from transformer_block import TransformerBlock

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




class MiniViT(nn.Module):
    """
    Minimal Vision Transformer for MNIST classification.

    Args:
        img_size   : 28 (MNIST)
        patch_size : 4  → 49 patches
        in_channels: 1  (greyscale)
        d_model    : embedding dimension
        n_heads    : attention heads
        n_layers   : number of Transformer blocks
        d_ff       : FFN inner dimension
        n_classes  : 10 (MNIST digits)
        dropout    : dropout probability
    """

    def __init__(
        self,
        img_size:    int = 28,
        patch_size:  int = 4,
        in_channels: int = 1,
        d_model:     int = 64,
        n_heads:     int = 8,
        n_layers:    int = 6,
        d_ff:        int = 256,
        n_classes:   int = 10,
        dropout:     float = 0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, d_model)
        n_patches = self.patch_embed.n_patches

        # [CLS] token — one learnable vector per model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learned position embeddings: n_patches + 1 (for CLS)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, d_model))

        self.pos_drop = nn.Dropout(dropout)

        # Stack of Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.head = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self):
        # Standard ViT initialisation
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.pos_embed,  std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 28, 28)
        returns: (batch, 10) logits
        """
        B = x.size(0)

        # 1. Patch embedding
        x = self.patch_embed(x)               # (B, 49, 64)

        # 2. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1) # (B, 1, 64)
        x   = torch.cat([cls, x], dim=1)       # (B, 50, 64)

        # 3. Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Transformer blocks
        x = self.blocks(x)                     # (B, 50, 64)

        # 5. LayerNorm + take CLS token
        x = self.norm(x)
        cls_out = x[:, 0]                      # (B, 64)

        # 6. Classify
        return self.head(cls_out)              # (B, 10)


# Instantiate and inspect
if __name__ == "__main__":
    print("""
  Full ViT pipeline:

    Image (B, 1, 28, 28)
      ↓  PatchEmbedding             → (B, 49, 64)
      ↓  Prepend CLS token          → (B, 50, 64)   [CLS] + 49 patches
      ↓  + Learned position embed   → (B, 50, 64)
      ↓  Dropout
      ↓  TransformerBlock |X| N     → (B, 50, 64)
      ↓  LayerNorm
      ↓  Take CLS token: x[:,0,:]   → (B, 64)
      ↓  MLP head: Linear(64, 10)   → (B, 10)    logits

  CLS token:
      A learned vector prepended to the patch sequence.
      After N attention blocks, it has "attended to" all patches.
      The final CLS representation summarises the whole image.
      The classifier reads only from CLS.

  This is identical to BERT's [CLS] token for sentence classification.
""")

    vit = MiniViT(
        img_size=28, patch_size=4, in_channels=1,
        d_model=64, n_heads=8, n_layers=6, d_ff=256,
        n_classes=10, dropout=0.1
    ).to(DEVICE)

    total_params = sum(p.numel() for p in vit.parameters())
    print(f"\n  MiniViT parameters: {total_params:,}")

    dummy = torch.randn(4, 1, 28, 28).to(DEVICE)
    with torch.no_grad():
        out = vit(dummy)
    print(f"  Input  shape: {dummy.shape}")
    print(f"  Output shape: {out.shape}  (logits, 10 classes) ✓")

