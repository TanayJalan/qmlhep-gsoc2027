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

print("SECTION 1 — Attention Formula Step by Step")

print("""
  Attention(Q, K, V) = softmax( Q @ K.T / sqrt(d_k) ) @ V

  Q: queries  — shape (batch, seq_len, d_k)   "what am I looking for?"
  K: keys     — shape (batch, seq_len, d_k)   "what do I contain?"
  V: values   — shape (batch, seq_len, d_v)   "what do I output?"

  Step 1: scores  = Q @ K.T          (batch, seq, seq)  raw similarities
  Step 2: scores /= sqrt(d_k)        scale to prevent softmax saturation
  Step 3: weights = softmax(scores)  normalise to probability distribution
  Step 4: output  = weights @ V      weighted sum of values

  Why divide by sqrt(d_k)?
    With d_k=64, each dot product is a sum of 64 products.
    The variance of Q@K.T grows with d_k — without scaling,
    large d_k pushes softmax inputs into the saturating regime
    where gradients vanish. Dividing by sqrt(d_k) keeps variance ≈ 1.
""")


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention.

    Args:
        Q    : (batch, heads, seq_q, d_k)
        K    : (batch, heads, seq_k, d_k)
        V    : (batch, heads, seq_k, d_v)
        mask : (batch, 1, 1, seq_k)  or  (batch, 1, seq_q, seq_k)
               Positions where mask==0 are set to -inf before softmax.
        dropout: optional Dropout applied to attention weights

    Returns:
        output  : (batch, heads, seq_q, d_v)
        weights : (batch, heads, seq_q, seq_k)  attention weight matrix
    """
    d_k = Q.size(-1)

    # Step 1 + 2: scaled scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (batch, heads, seq_q, seq_k)

    # Step 3a: apply mask (for padding or causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3b: softmax over key dimension
    weights = F.softmax(scores, dim=-1)

    # Handle all-masked rows (produces NaN from softmax(-inf)) → set to 0
    weights = torch.nan_to_num(weights, nan=0.0)

    # Optional dropout on attention weights
    if dropout is not None:
        weights = dropout(weights)

    # Step 4: weighted sum of values
    output = torch.matmul(weights, V)
    # output: (batch, heads, seq_q, d_v)

    return output, weights


# ── Quick test (single head, no batch) ───────────────────────
print("\n  Single-head attention on a toy sequence:")

batch, seq, d_k, d_v = 1, 5, 8, 8
Q = torch.randn(batch, 1, seq, d_k)
K = torch.randn(batch, 1, seq, d_k)
V = torch.randn(batch, 1, seq, d_v)

out, attn = scaled_dot_product_attention(Q, K, V)
print(f"  Q shape:       {Q.shape}")
print(f"  Output shape:  {out.shape}")
print(f"  Weights shape: {attn.shape}")
print(f"  Weight rows sum to 1: "
      f"{torch.allclose(attn.sum(-1), torch.ones(batch, 1, seq))} ✓")


print("SECTION 2 — Padding Mask")

print("""
  Real sequences in a batch have different lengths.
  Shorter ones are padded with zeros to match the longest.
  We must prevent attention from attending to padding tokens —
  otherwise the model learns patterns in meaningless zeros.

  Mask: 1 = real token (attend), 0 = padding (ignore → -inf)
""")

batch, seq = 2, 6
d_k = 16

# Sequence lengths in this batch: [4, 6]
pad_mask_np = torch.tensor([
    [1, 1, 1, 1, 0, 0],   # 4 real tokens, 2 padding
    [1, 1, 1, 1, 1, 1],   # 6 real tokens, no padding
], dtype=torch.float32)

# Reshape for broadcasting: (batch, 1, 1, seq_k)
pad_mask = pad_mask_np.unsqueeze(1).unsqueeze(2)

Q = torch.randn(batch, 1, seq, d_k)
K = torch.randn(batch, 1, seq, d_k)
V = torch.randn(batch, 1, seq, d_k)

out, attn = scaled_dot_product_attention(Q, K, V, mask=pad_mask)

print(f"\n  Batch 1 attention weights (rows should sum to 1):")
print(f"  Last 2 columns should be 0 (padding positions):")
print(f"  {attn[0, 0].round(decimals=3)}")

print(f"\n  Batch 2 attention weights (no padding — all non-zero):")
print(f"  {attn[1, 0].round(decimals=3)}")

# Verify padding positions have zero weight
padding_attn = attn[0, 0, :, 4:]   # last 2 columns of batch 0
print(f"\n  Padding positions weight (should be 0): {padding_attn.round(decimals=6)}")
assert torch.allclose(padding_attn, torch.zeros_like(padding_attn))
print(f"Padding tokens receive zero attention weight")


print("SECTION 3 — Causal Mask (for GPT-style decoders)")

print("""
  In a language model, token i should only attend to tokens 0..i.
  Attending to future tokens would be "cheating" during training
  because at inference time, future tokens don't exist yet.

  The causal mask is a lower-triangular matrix:
      mask[i,j] = 1 if j <= i else 0

  ViT (our Phase 3 target) does NOT use this — it's bidirectional.
  But understanding it is essential for GSoC Task VIII's written design,
  where you explain how a QVIT would be extended to sequence tasks.
""")

seq = 5
causal_mask = torch.tril(torch.ones(seq, seq))
print(f"\n Causal mask (seq={seq}):")
print(f"{causal_mask.int()}")

Q = torch.randn(1, 1, seq, 8)
K = torch.randn(1, 1, seq, 8)
V = torch.randn(1, 1, seq, 8)

mask_4d = causal_mask.unsqueeze(0).unsqueeze(0)   # (1, 1, seq, seq)
out, attn = scaled_dot_product_attention(Q, K, V, mask=mask_4d)

print(f"\n  Attention weights with causal mask:")
print(f"(upper triangle should be 0 — future positions blocked)")
print(f"{attn[0, 0].round(decimals=3)}")

upper_tri = attn[0, 0].triu(diagonal=1)
assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-6)
print(f"Upper triangle is zero — causal masking works")

print("SECTION 4 — Attention Patterns")

print("""
  Different Q/K configurations produce interpretably different patterns.
  This is useful for debugging and for the Task VIII design write-up.
""")

seq, d_k = 8, 32

# Pattern 1: Identity attention — each token attends only to itself
Q_id = torch.eye(seq).unsqueeze(0).unsqueeze(0).expand(1, 1, seq, seq)
K_id = Q_id.clone()
V_id = torch.randn(1, 1, seq, d_k)
_, attn_id = scaled_dot_product_attention(
    Q_id[..., :d_k], K_id[..., :d_k], V_id)
print(f"\n  Near-identity attention (diag ≈ 1, off-diag ≈ 0):")
diag_mean = attn_id[0, 0].diag().mean().item()
print(f"  Diagonal mean: {diag_mean:.4f}  (high = attending to self)")

# Pattern 2: Uniform attention — fully uninformative
Q_u = torch.zeros(1, 1, seq, d_k)
K_u = torch.zeros(1, 1, seq, d_k)
V_u = torch.randn(1, 1, seq, d_k)
_, attn_u = scaled_dot_product_attention(Q_u, K_u, V_u)
print(f"\n  Uniform attention (all weights ≈ 1/seq):")
print(f"First row: {attn_u[0, 0, 0].round(decimals=4).tolist()}")
expected_uniform = 1.0 / seq
print(f"Expected: {[round(expected_uniform, 4)] * seq}")
assert torch.allclose(attn_u, torch.ones_like(attn_u) / seq, atol=1e-5)
print(f"Uniform attention confirmed")


print("SECTION 5 — Verify Against F.scaled_dot_product_attention")

print("""
  PyTorch 2.0+ has F.scaled_dot_product_attention built in.
  It uses FlashAttention under the hood (faster, lower memory).
  Our implementation should give identical results.
""")

batch, heads, seq, d_k = 2, 4, 10, 16
Q = torch.randn(batch, heads, seq, d_k)
K = torch.randn(batch, heads, seq, d_k)
V = torch.randn(batch, heads, seq, d_k)

# Ours
out_ours, _ = scaled_dot_product_attention(Q, K, V)

# PyTorch built-in
out_pt = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)

match = torch.allclose(out_ours, out_pt, atol=1e-5)
print(f"\n  Our output shape:    {out_ours.shape}")
print(f"  PyTorch output shape:{out_pt.shape}")
print(f"  Outputs match: {'✓' if match else '✗'}")
print(f"  Max absolute diff: {(out_ours - out_pt).abs().max().item():.2e}")