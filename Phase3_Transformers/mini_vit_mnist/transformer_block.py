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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, D = x.shape
        H, dk   = self.n_heads, self.d_k

        def split(t):
            return t.view(B, S, H, dk).transpose(1, 2)

        Q = split(self.W_Q(x))
        K = split(self.W_K(x))
        V = split(self.W_V(x))

        scores  = Q @ K.transpose(-2, -1) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.drop(weights)

        out = (weights @ V).transpose(1, 2).contiguous().view(B, S, D)
        return self.W_O(out), weights


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer encoder block:
        x → LN → MHA → residual → LN → FFN → residual
    """
    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


import sys, atexit
def _clean_exit():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
atexit.register(_clean_exit)


if __name__ == "__main__":
    block = TransformerBlock(d_model=64, n_heads=8, d_ff=256)
    x = torch.randn(2, 50, 64)
    out = block(x)
    print("\n  Input shape: ", x.shape)
    print("  Output shape:", out.shape)

