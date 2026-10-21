
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):

    def __init__(self, img_size: int = 28, patch_size: int = 4,
                 in_channels: int = 1, d_model: int = 64):
        super().__init__()
        assert img_size % patch_size == 0, \
            f"img_size {img_size} must be divisible by patch_size {patch_size}"

        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2   # 7×7 = 49

        self.projection = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)     # (B, d_model, H/p, W/p)
        x = x.flatten(2)           # (B, d_model, n_patches)
        return x.transpose(1, 2)   # (B, n_patches, d_model)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention (Q=K=V=x).
    Uses nn.MultiheadAttention internally (batch_first=True).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return self.proj_drop(out)

class FeedForward(nn.Module):
    """
    Position-wise FFN: Linear → GELU → Dropout → Linear → Dropout
    d_ff is typically 4 × d_model.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer encoder block:
        x → LN → MHA → residual → LN → FFN → residual

    Pre-LN trains more stably than Post-LN for deep models.
    Used in ViT and most modern Transformer variants.
    """

    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn   = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class MiniViT(nn.Module):
    def __init__(
        self,
        img_size:    int   = 28,
        patch_size:  int   = 4,
        in_channels: int   = 1,
        d_model:     int   = 64,
        n_heads:     int   = 8,
        n_layers:    int   = 6,
        d_ff:        int   = 256,
        n_classes:   int   = 10,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, d_model)
        n_patches = self.patch_embed.n_patches

        # 2. CLS token (one learnable vector per model instance)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 3. Learned positional embeddings (CLS + n_patches positions)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, d_model))

        self.pos_drop = nn.Dropout(dropout)

        # 4. Transformer encoder blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 5. Final normalisation
        self.norm = nn.LayerNorm(d_model)

        # 6. Classification head (reads CLS token only)
        self.head = nn.Linear(d_model, n_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token,   std=0.02)
        nn.init.trunc_normal_(self.pos_embed,   std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, 28, 28)
        returns: (batch, 10) raw logits
        """
        B = x.size(0)

        # Patch embed: (B, 1, 28, 28) → (B, 49, 64)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]                   
        return self.head(x)           

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            'patch_embed': sum(p.numel()
                               for p in self.patch_embed.parameters()),
            'pos_embed':   self.pos_embed.numel() + self.cls_token.numel(),
            'transformer': sum(p.numel() for p in self.blocks.parameters()),
            'head':        sum(p.numel() for p in self.head.parameters()),
            'total':       total,
        }

if __name__ == "__main__":
    model  = MiniViT()
    params = model.count_parameters()

    print("MiniViT parameter breakdown:")
    for k, v in params.items():
        print(f"  {k:<14}: {v:>8,}")

    dummy = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        out = model(dummy)

    print(f"\nInput:  {dummy.shape}")
    print(f"Output: {out.shape}  (logits, 10 classes)")
    assert out.shape == (4, 10)
    print("Sanity check passed ok")
