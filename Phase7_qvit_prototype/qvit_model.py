import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pennylane as qml
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import time
import multiprocessing

torch.manual_seed(42)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"  Device: {DEVICE}")
print(f"  PennyLane: {qml.__version__}")

try:
    from quantum_attention import QuantumAttentionHead, N_QUBITS, N_QLAYERS
except ImportError:
    from Phase7_qvit_prototype.quantum_attention import QuantumAttentionHead, N_QUBITS, N_QLAYERS


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channels=1, d_model=64):
        super().__init__()
        self.n_patches  = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, d_model,
                                     kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)      # (B, d_model, h, w)
        x = x.flatten(2)            # (B, d_model, n_patches)
        return x.transpose(1, 2)    # (B, n_patches, d_model)


class QuantumMHA(nn.Module):
    """
    Quantum Multi-Head Attention.
    Replaces W_Q, W_K, W_V with QuantumAttentionHead instances.
    W_O projection stays classical.
    """
    def __init__(self, d_model: int, n_heads: int,
                 n_qubits: int = N_QUBITS, n_qlayers: int = N_QLAYERS,
                 dropout: float = 0.1):
        super().__init__()
        self.n_heads  = n_heads
        self.n_qubits = n_qubits
        self.head_dim = n_qubits   # each head outputs n_qubits values

        # Quantum projections for Q, K, V
        self.q_heads = nn.ModuleList([
            QuantumAttentionHead(d_model, n_qubits, n_qlayers)
            for _ in range(n_heads)
        ])
        self.k_heads = nn.ModuleList([
            QuantumAttentionHead(d_model, n_qubits, n_qlayers)
            for _ in range(n_heads)
        ])
        self.v_heads = nn.ModuleList([
            QuantumAttentionHead(d_model, n_qubits, n_qlayers)
            for _ in range(n_heads)
        ])

        # Classical output projection
        self.W_O    = nn.Linear(n_heads * n_qubits, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model)"""
        head_outputs = []
        for h in range(self.n_heads):
            Q = self.q_heads[h](x)   # (B, S, n_qubits)
            K = self.k_heads[h](x)
            V = self.v_heads[h](x)

            # Scaled dot-product attention
            scale   = math.sqrt(self.n_qubits)
            scores  = torch.matmul(Q, K.transpose(-2, -1)) / scale
            weights = F.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            head_out = torch.matmul(weights, V)   # (B, S, n_qubits)
            head_outputs.append(head_out)

        # Concatenate heads and project
        concat = torch.cat(head_outputs, dim=-1)  # (B, S, n_heads*n_qubits)
        return self.W_O(concat)                   # (B, S, d_model)


class QViTBlock(nn.Module):
    """Quantum Transformer block (Pre-LN)."""
    def __init__(self, d_model, n_heads, d_ff, n_qubits, n_qlayers, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = QuantumMHA(d_model, n_heads, n_qubits, n_qlayers, dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MiniQViT(nn.Module):
    """
    Minimal Quantum Vision Transformer for MNIST.

    Kept small for simulator feasibility:
        n_heads  = 2   (2 quantum attention heads)
        n_blocks = 2   (2 QVIT blocks)
        d_model  = 64
        n_qubits = 4   (per head)
    """
    def __init__(self, img_size=28, patch_size=4, in_channels=1,
                 d_model=64, n_heads=2, n_blocks=2, d_ff=128,
                 n_qubits=N_QUBITS, n_qlayers=N_QLAYERS,
                 n_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                           in_channels, d_model)
        n_patches        = (img_size // patch_size) ** 2

        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self.pos_drop    = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            QViTBlock(d_model, n_heads, d_ff, n_qubits, n_qlayers, dropout)
            for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pos_drop(x + self.pos_embed)
        x   = self.blocks(x)
        x   = self.norm(x)
        return self.head(x[:, 0])


if __name__ == "__main__":
    # Instantiate and inspect
    qvit = MiniQViT(n_heads=2, n_blocks=2, n_qubits=N_QUBITS, n_qlayers=N_QLAYERS)

    total_params   = sum(p.numel() for p in qvit.parameters())
    quantum_params = sum(p.numel() for n, p in qvit.named_parameters()
                         if 'weights' in n)
    classical_params = total_params - quantum_params

    print(f"\n  MiniQViT parameters:")
    print(f"    Total:     {total_params:,}")
    print(f"    Quantum:   {quantum_params:,}  "
          f"({100*quantum_params/total_params:.1f}%)")
    print(f"    Classical: {classical_params:,}")

    dummy = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        out = qvit(dummy)
    print(f"\n  Input:  {dummy.shape}")
    print(f"  Output: {out.shape}  (logits, 10 classes) ok")

