import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import atexit

def _clean_exit():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
atexit.register(_clean_exit)

torch.manual_seed(0)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k     = Q.size(-1)
    scores  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return torch.matmul(weights, V), weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads

        # Learned projection matrices
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        # Xavier init
        for layer in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(layer.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch, seq, d_model) → (batch, n_heads, seq, d_k)
        by viewing d_model as (n_heads, d_k) and transposing.
        """
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.n_heads, self.d_k)
        return x.transpose(1, 2)   # (batch, n_heads, seq, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverse of _split_heads:
        (batch, n_heads, seq, d_k) → (batch, seq, d_model)
        """
        batch, _, seq, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq, self.d_model)

    def forward(self,
                query: torch.Tensor,
                key:   torch.Tensor,
                value: torch.Tensor,
                mask:  torch.Tensor = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query : (batch, seq_q, d_model)
            key   : (batch, seq_k, d_model)
            value : (batch, seq_k, d_model)
            mask  : optional padding or causal mask

        Returns:
            output  : (batch, seq_q, d_model)
            weights : (batch, n_heads, seq_q, seq_k)
        """
        # Project and split into heads
        Q = self._split_heads(self.W_Q(query))   # (B, H, seq_q, d_k)
        K = self._split_heads(self.W_K(key))      # (B, H, seq_k, d_k)
        V = self._split_heads(self.W_V(value))    # (B, H, seq_k, d_k)

        # Attend
        attn_out, weights = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = self.dropout(attn_out)

        # Merge heads and project
        merged = self._merge_heads(attn_out)      # (B, seq_q, d_model)
        output = self.W_O(merged)                 # (B, seq_q, d_model)

        return output, weights


# ── Test ─────────────────────────────────────────────────────
batch, seq, d_model, n_heads = 2, 10, 64, 8
mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

x   = torch.randn(batch, seq, d_model)
out, attn_weights = mha(x, x, x)    # self-attention: Q=K=V=x

print(f"\n  Input  shape: {x.shape}")
print(f"  Output shape: {out.shape}")
print(f"  Attn weights: {attn_weights.shape}  (batch, heads, seq, seq)")
print(f"  Weight rows sum to 1: "
      f"{torch.allclose(attn_weights.sum(-1), torch.ones(batch, n_heads, seq))} ✓")

# Parameter count
mha_params = sum(p.numel() for p in mha.parameters())
print(f"\n  MHA parameters: {mha_params:,}")
print(f"  = 4 x d_model² = 4 x {d_model}² = {4 * d_model**2:,}")

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding.
    Added to token embeddings before the first Transformer block.
    """
    def __init__(self, d_model: int, max_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build the encoding matrix once and register as buffer
        # (buffer: saved with model but not a trainable parameter)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()   # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)   # even dims
        pe[:, 1::2] = torch.cos(pos * div)   # odd dims

        pe = pe.unsqueeze(0)                  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings — used in ViT.
    One embedding vector per position, updated by gradient descent.
    """
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout   = nn.Dropout(p=dropout)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model)"""
        seq  = x.size(1)
        pos  = torch.arange(seq, device=x.device).unsqueeze(0)  # (1, seq)
        x    = x + self.embedding(pos)
        return self.dropout(x)


# Test both
batch, seq, d_model = 2, 16, 64

sin_pe  = SinusoidalPositionalEncoding(d_model)
learned = LearnedPositionalEncoding(d_model, max_len=100)

x = torch.randn(batch, seq, d_model)
print(f"\n  Input shape:                  {x.shape}")
print(f"  After sinusoidal PE:          {sin_pe(x).shape}")
print(f"  After learned PE:             {learned(x).shape}")
print(f"\n  Sinusoidal PE parameters:    0  (fixed)")
print(f"  Learned PE parameters:        "
      f"{sum(p.numel() for p in learned.parameters()):,}")

# Show PE pattern
pe_matrix = sin_pe.pe[0, :8, :8]
print(f"\n  First 8 positions × 8 dims of sinusoidal PE:")
print(f"  {pe_matrix.round(decimals=3)}")

class FeedForward(nn.Module):
    """Position-wise FFN: two linear layers with GELU activation."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),                  # ViT uses GELU, not ReLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """
    One Transformer encoder block (Pre-LN variant used in ViT):

        x → LN → MHA → residual → LN → FFN → residual

    Pre-LN (LayerNorm before sublayer) trains more stably than
    the original Post-LN (LayerNorm after residual).
    """
    def __init__(self, d_model: int, n_heads: int,
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn   = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention sublayer with residual
        attn_out, _ = self.attn(self.norm1(x),
                                self.norm1(x),
                                self.norm1(x), mask)
        x = x + attn_out

        # FFN sublayer with residual
        x = x + self.ffn(self.norm2(x))
        return x


# Test
d_model, n_heads, d_ff = 64, 8, 256
block = TransformerEncoderBlock(d_model, n_heads, d_ff)

x   = torch.randn(2, 10, d_model)
out = block(x)
print(f"\n  Input shape:  {x.shape}")
print(f"  Output shape: {out.shape}  (same — encoder preserves shape)")

block_params = sum(p.numel() for p in block.parameters())
print(f"  Block parameters: {block_params:,}")
print(f"  Breakdown:")
mha_p = sum(p.numel() for p in block.attn.parameters())
ffn_p = sum(p.numel() for p in block.ffn.parameters())
ln_p  = sum(p.numel() for p in block.norm1.parameters()) * 2
print(f"    MHA:       {mha_p:,}")
print(f"    FFN:       {ffn_p:,}")
print(f"    LayerNorm: {ln_p:,}")

batch, seq, d_model, n_heads = 2, 6, 32, 4

our_mha = MultiHeadAttention(d_model, n_heads, dropout=0.0)
pt_mha  = nn.MultiheadAttention(d_model, n_heads,
                                  dropout=0.0, batch_first=True,
                                  bias=False)

# Copy our weights into PyTorch's format
# PyTorch stores W_Q, W_K, W_V concatenated as in_proj_weight
with torch.no_grad():
    pt_mha.in_proj_weight.copy_(
        torch.cat([our_mha.W_Q.weight,
                   our_mha.W_K.weight,
                   our_mha.W_V.weight], dim=0)
    )
    pt_mha.out_proj.weight.copy_(our_mha.W_O.weight)

x = torch.randn(batch, seq, d_model)

our_out, _  = our_mha(x, x, x)
pt_out,  _  = pt_mha(x, x, x)

match = torch.allclose(our_out, pt_out, atol=1e-5)
print(f"\n  Our MHA output shape: {our_out.shape}")
print(f"  PyTorch MHA shape:    {pt_out.shape}")
print(f"  Outputs match: {'✓' if match else '✗'}")
print(f"  Max diff: {(our_out - pt_out).abs().max().item():.2e}")
