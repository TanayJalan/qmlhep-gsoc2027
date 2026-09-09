"""
Positional Encoding — giving a Transformer its sense of order.

Attention is permutation-equivariant: if you shuffle the input sequence,
the output shuffles identically. The model has no concept of position.
Positional encoding fixes this by adding position information to each
token embedding before the first Transformer block.

Covers:
    1. Why positional encoding is needed (permutation invariance demo)
    2. Sinusoidal PE (original Transformer, fixed, no parameters)
    3. Learned PE (ViT, BERT — trainable lookup table)
    4. Relative PE (Shaw et al. 2018 — encodes distance, not absolute position)
    5. Rotary PE / RoPE (used in LLaMA, GPT-NeoX — modern standard)
    6. Visualising and comparing all encodings
    7. Which to use for your mini-ViT

"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

torch.manual_seed(0)


class BareAttention(nn.Module):
    """Minimal self-attention with no positional info."""
    def __init__(self, d_model):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        q = k = v = self.W(x)
        scores  = q @ k.transpose(-2, -1) / math.sqrt(x.size(-1))
        weights = torch.softmax(scores, dim=-1)
        return weights @ v

d_model = 16
attn    = BareAttention(d_model)

# Original sequence
x_orig  = torch.randn(1, 5, d_model)

# Shuffled sequence: swap tokens 1 and 3
perm    = [0, 3, 2, 1, 4]
x_shuf  = x_orig[:, perm, :]

with torch.no_grad():
    out_orig = attn(x_orig)
    out_shuf = attn(x_shuf)

# Output of shuffled input should equal shuffled output of original
out_orig_shuf = out_orig[:, perm, :]
perm_equivariant = torch.allclose(out_shuf, out_orig_shuf, atol=1e-5)

print(f"\n Permutation equivariance test:")
print(f"attn(shuffle(X)) == shuffle(attn(X)): "
      f"{'TRUE' if perm_equivariant else 'FALSE'}")
print(f"\n -> The model is blind to token order.")
print(f" -> Positional encoding breaks this symmetry.")

print("""
  Formula:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

  Intuition:
      Low dimensions  → high frequency (changes fast across positions)
      High dimensions → low frequency (changes slowly)
      Like a binary counter but in continuous sinusoids.
""")
class SinusoidalPE(nn.Module):
    """
    Fixed sinusoidal positional encoding.
    Registered as a buffer — saved with the model but not trained.
    """
    def __init__(self, d_model: int, max_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)           # (max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )                                              # (d_model/2,)

        pe[:, 0::2] = torch.sin(pos * div)            # even indices
        pe[:, 1::2] = torch.cos(pos * div)            # odd indices

        pe = pe.unsqueeze(0)                           # (1, max_len, d_model)
        self.register_buffer('pe', pe)                 # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model)"""
        return self.dropout(x + self.pe[:, :x.size(1)])

    def get_matrix(self, seq_len: int) -> torch.Tensor:
        """Return PE matrix for inspection: (seq_len, d_model)."""
        return self.pe[0, :seq_len]


sin_pe = SinusoidalPE(d_model=64, max_len=100)
pe_mat = sin_pe.get_matrix(seq_len=20)

print(f"\n  PE matrix shape:   {pe_mat.shape}  (seq_len=20, d_model=64)")
print(f"  Parameters:        0  (fixed, not trainable)")

# Test forward
x = torch.randn(2, 20, 64)
out = sin_pe(x)
print(f"  Input  shape:      {x.shape}")
print(f"  Output shape:      {out.shape}")

# Show first few values
print(f"\n  PE values for first 5 positions, first 8 dims:")
print(f"  {'pos':>4}  " + "  ".join([f"d{i:02d}" for i in range(8)]))
for p in range(5):
    vals = pe_mat[p, :8].tolist()
    print(f"  {p:>4}  " + "  ".join([f"{v:+.3f}" for v in vals]))

# Key property: PE difference encodes relative distance
print(f"\n  Distance property check (PE[5] - PE[3] ≈ PE[2] - PE[0]):")
diff_53 = (pe_mat[5] - pe_mat[3]).norm().item()
diff_20 = (pe_mat[2] - pe_mat[0]).norm().item()
print(f"  ||PE[5]-PE[3]|| = {diff_53:.4f}")
print(f"  ||PE[2]-PE[0]|| = {diff_20:.4f}")
print(f"  (Same distance apart → similar PE difference ✓)")


class LearnedPE(nn.Module):
    """
    Learned positional embeddings.
    One embedding vector per position, updated by gradient descent.
    """
    def __init__(self, d_model: int, max_len: int,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout   = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model)"""
        seq = x.size(1)
        pos = torch.arange(seq, device=x.device).unsqueeze(0)  # (1, seq)
        return self.dropout(x + self.embedding(pos))

    def get_matrix(self, seq_len: int) -> torch.Tensor:
        pos = torch.arange(seq_len)
        return self.embedding(pos).detach()


# For ViT-MNIST: 49 patches + 1 CLS token = 50 positions
learned_pe = LearnedPE(d_model=64, max_len=50)
pe_learned = learned_pe.get_matrix(50)

total_params = sum(p.numel() for p in learned_pe.parameters())
print(f"\n  Learned PE for ViT-MNIST:")
print(f"  max_len = 50  (49 patches + 1 CLS)")
print(f"  d_model = 64")
print(f"  Parameters: {total_params:,}  (50 × 64 = {50*64})")

x = torch.randn(2, 50, 64)
out = learned_pe(x)
print(f"\n  Input  shape: {x.shape}")
print(f"  Output shape: {out.shape}")

# Compare initial similarity between positions
# After training, similar positions should have similar embeddings
pe_init = learned_pe.get_matrix(10)
sim_matrix = pe_init @ pe_init.T
print(f"\n  Initial cosine similarity between positions (before training):")
print(f"  (After training, nearby positions become more similar)")
norms = pe_init.norm(dim=1, keepdim=True)
cos_sim = (pe_init @ pe_init.T) / (norms @ norms.T)
print(f"  {cos_sim[:5, :5].round(decimals=2)}")

class RelativePE(nn.Module):
    """
    Simplified relative positional embedding.
    Adds a learnable bias to attention scores based on distance.
    """
    def __init__(self, d_model: int, max_dist: int = 32):
        super().__init__()
        self.max_dist   = max_dist
        # Embedding for each relative distance from -max_dist to +max_dist
        self.rel_embed  = nn.Embedding(2 * max_dist + 1, d_model)
        nn.init.trunc_normal_(self.rel_embed.weight, std=0.02)

    def get_bias(self, seq_len: int, device) -> torch.Tensor:
        """
        Compute (seq_len, seq_len) matrix of relative position biases.
        bias[i,j] = embedding of (j - i), clipped to [-max_dist, max_dist]
        """
        positions = torch.arange(seq_len, device=device)
        distances = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq, seq)
        distances = distances.clamp(-self.max_dist, self.max_dist)
        distances = distances + self.max_dist                          # shift to [0, 2*max]
        return distances                                               # (seq, seq)


rel_pe = RelativePE(d_model=64, max_dist=16)
bias   = rel_pe.get_bias(seq_len=10, device='cpu')
print(f"\n  Relative distance matrix (10 × 10):")
print(f"  (values are distance + max_dist, centred at 16)")
print(f"  {bias}")
print(f"\n  Embedding matrix shape: {rel_pe.rel_embed.weight.shape}")
print(f"  Parameters: {sum(p.numel() for p in rel_pe.parameters()):,}")

def apply_rope(x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
    """
    Apply Rotary Position Embedding to tensor x.
    x: (batch, seq, d_model) — d_model must be even

    Rotates pairs of dimensions by position-dependent angles.
    """
    batch, seq, d = x.shape
    assert d % 2 == 0

    # Rotation angles: θ_i = 10000^(-2i/d) for i = 0..d/2-1
    i      = torch.arange(0, d // 2, dtype=torch.float32)
    theta  = 10000.0 ** (-2 * i / d)                       # (d/2,)
    pos    = torch.arange(seq, dtype=torch.float32)         # (seq,)
    angles = torch.outer(pos, theta)                        # (seq, d/2)

    cos = torch.cos(angles).unsqueeze(0)   # (1, seq, d/2)
    sin = torch.sin(angles).unsqueeze(0)   # (1, seq, d/2)

    # Split x into pairs of dimensions
    x1 = x[..., 0::2]   # (batch, seq, d/2)  even dims
    x2 = x[..., 1::2]   # (batch, seq, d/2)  odd dims

    # Apply 2D rotation to each pair
    x_rot = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

    return x_rot.flatten(-2)   # (batch, seq, d)


# Test RoPE
x_test = torch.randn(2, 10, 64)
x_rope  = apply_rope(x_test)
print(f"\n  Input  shape: {x_test.shape}")
print(f"  After RoPE:   {x_rope.shape}  (shape preserved)")
print(f"  Parameters:   0  (no learned params)")

# Key property: relative position encoded in dot product
q = torch.randn(1, 8, 32)
k = torch.randn(1, 8, 32)

q_rope = apply_rope(q)
k_rope = apply_rope(k)

# Score at position (i=2, j=5) — distance 3
score_rope = (q_rope[0, 2] @ k_rope[0, 5]).item()
score_base = (q[0, 2] @ k[0, 5]).item()
print(f"\n  Base score q[2]·k[5]:  {score_base:.4f}")
print(f"  RoPE score q[2]·k[5]:  {score_rope:.4f}")
print(f"  (RoPE encodes that tokens are 3 apart in the score)")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Sinusoidal PE heatmap
sin_pe_vis = SinusoidalPE(d_model=64, dropout=0.0)
pe_sin_mat = sin_pe_vis.get_matrix(50).numpy()
im1 = axes[0].imshow(pe_sin_mat, aspect='auto', cmap='RdBu',
                      vmin=-1, vmax=1)
axes[0].set_title('Sinusoidal PE\n(50 positions × 64 dims)')
axes[0].set_xlabel('Dimension')
axes[0].set_ylabel('Position')
plt.colorbar(im1, ax=axes[0])

# Plot 2: Learned PE similarity matrix (after init)
learned_vis = LearnedPE(d_model=64, max_len=50, dropout=0.0)
pe_l_mat    = learned_vis.get_matrix(50).numpy()
norms_vis   = np.linalg.norm(pe_l_mat, axis=1, keepdims=True)
cos_vis     = pe_l_mat @ pe_l_mat.T / (norms_vis @ norms_vis.T + 1e-8)
im2 = axes[1].imshow(cos_vis, aspect='auto', cmap='RdBu',
                      vmin=-1, vmax=1)
axes[1].set_title('Learned PE Cosine Similarity\n(50×50, before training)')
axes[1].set_xlabel('Position j')
axes[1].set_ylabel('Position i')
plt.colorbar(im2, ax=axes[1])

# Plot 3: Sinusoidal PE for first 4 dimensions vs position
positions = np.arange(50)
for dim in range(4):
    axes[2].plot(positions, pe_sin_mat[:, dim],
                 label=f'dim {dim}', linewidth=1.5)
axes[2].set_title('Sinusoidal PE\nFirst 4 dims vs position')
axes[2].set_xlabel('Position')
axes[2].set_ylabel('PE value')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__))
                         if '__file__' in dir() else '.', 'pe_visualisation.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n  Plot saved to: {out_path}")
plt.close()

