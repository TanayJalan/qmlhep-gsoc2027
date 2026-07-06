"""
Phase 1 — Week 1, Day 1-2
File: phase1_foundations/numpy_basics/02_vectorization.py

Topics covered:
    For each operation below, we implement it 3 ways and time all three:
        1. Pure Python loop
        2. NumPy vectorized
        3. NumPy with broadcasting
    Then print the speedup factor so the difference is visceral, not abstract.

Operations covered:
    1. Element-wise multiply two arrays          (dot product warmup)
    2. Row-wise mean subtraction                 (feature normalisation)
    3. Pairwise dot products between two sets    (attention score prototype)
    4. ReLU activation                           (neural net activation)
    5. Softmax over a batch                      (classifier output layer)

Run with:
    python 02_vectorization.py

Expected output: loops are 50x–500x slower than NumPy.
That number is why ML is written in NumPy/PyTorch, not pure Python.
"""

import numpy as np
import time

# ─────────────────────────────────────────────────────────────
# Helper: run a function N times and return median wall time
# ─────────────────────────────────────────────────────────────
def benchmark(fn, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times), result


def print_header(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


def print_timing(label, t, baseline=None):
    msg = f"  {label:<30} {t*1000:>8.3f} ms"
    if baseline is not None:
        speedup = baseline / t
        msg += f"   ({speedup:.1f}x faster)"
    print(msg)


# ─────────────────────────────────────────────────────────────
# OPERATION 1 — Element-wise multiply (dot product warmup)
# ─────────────────────────────────────────────────────────────
print_header("OPERATION 1 — Element-wise Multiply")
print("  Task: multiply two arrays of 1,000,000 floats element-wise")
print("  ML use: scaling activations, applying masks, weight updates\n")

N = 1_000_000
a = np.random.randn(N)
b = np.random.randn(N)

# Method 1 — Pure Python loop
def loop_multiply():
    result = [0.0] * N
    for i in range(N):
        result[i] = a[i] * b[i]
    return result

# Method 2 — NumPy vectorized
def numpy_multiply():
    return a * b

# Method 3 — Broadcasting (same result, explicit axis notation)
def broadcast_multiply():
    return np.multiply(a, b)

t_loop, _      = benchmark(loop_multiply, repeats=3)
t_numpy, r_np  = benchmark(numpy_multiply)
t_bcast, r_bc  = benchmark(broadcast_multiply)

print_timing("Pure Python loop",   t_loop)
print_timing("NumPy vectorized",   t_numpy,  baseline=t_loop)
print_timing("NumPy broadcasting", t_bcast,  baseline=t_loop)

# Correctness check
print(f"\n  Results match: {np.allclose(r_np, r_bc)}")


# ─────────────────────────────────────────────────────────────
# OPERATION 2 — Row-wise mean subtraction (feature normalisation)
# ─────────────────────────────────────────────────────────────
print_header("OPERATION 2 — Row-wise Mean Subtraction")
print("  Task: subtract each row's mean from that row")
print("  Matrix: 10,000 samples × 512 features")
print("  ML use: batch normalisation, data preprocessing, LayerNorm\n")

ROWS, COLS = 10_000, 512
M = np.random.randn(ROWS, COLS)

# Method 1 — Pure Python loop over rows
def loop_row_mean():
    result = np.empty_like(M)
    for i in range(ROWS):
        row_mean = sum(M[i]) / COLS
        for j in range(COLS):
            result[i, j] = M[i, j] - row_mean
    return result

# Method 2 — NumPy vectorized (explicit loop replaced by axis kwarg)
def numpy_row_mean():
    row_means = np.mean(M, axis=1)       # shape: (10000,)
    # Need to reshape so subtraction broadcasts correctly
    return M - row_means.reshape(-1, 1)  # reshape to (10000, 1)

# Method 3 — Broadcasting with keepdims (cleaner, same speed)
def broadcast_row_mean():
    row_means = M.mean(axis=1, keepdims=True)  # shape: (10000, 1)
    return M - row_means                        # broadcasts over 512 columns

t_loop, _       = benchmark(loop_row_mean,    repeats=1)   # slow, run once
t_numpy, r_np   = benchmark(numpy_row_mean)
t_bcast, r_bc   = benchmark(broadcast_row_mean)

print_timing("Pure Python loop",           t_loop)
print_timing("NumPy vectorized",           t_numpy,  baseline=t_loop)
print_timing("Broadcasting + keepdims",    t_bcast,  baseline=t_loop)

print(f"\n  Results match: {np.allclose(r_np, r_bc)}")
print(f"  Row means after subtraction (should be ≈ 0):")
print(f"    max |mean|: {np.abs(r_bc.mean(axis=1)).max():.2e}")

print("""
  Note: keepdims=True is the idiomatic NumPy style.
  The reshape(-1, 1) version is equivalent but more verbose.
  You'll use keepdims constantly in transformer implementations.
""")


# ─────────────────────────────────────────────────────────────
# OPERATION 3 — Pairwise dot products (attention score prototype)
# ─────────────────────────────────────────────────────────────
print_header("OPERATION 3 — Pairwise Dot Products")
print("  Task: compute dot product of every query with every key")
print("  Queries: 64 vectors of dim 128   (Q matrix)")
print("  Keys:    64 vectors of dim 128   (K matrix)")
print("  Output:  64 × 64 score matrix    (attention logits)")
print("  ML use: THIS IS the core of self-attention in Transformers\n")

T, D = 64, 128                          # sequence length, dimension
Q = np.random.randn(T, D)
K = np.random.randn(T, D)

# Method 1 — Pure Python: double loop
def loop_dot():
    scores = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            scores[i, j] = sum(Q[i, k] * K[j, k] for k in range(D))
    return scores

# Method 2 — NumPy vectorized with np.dot
def numpy_dot():
    return np.dot(Q, K.T)              # (T, D) @ (D, T) → (T, T)

# Method 3 — Broadcasting via einsum (the Transformer notation)
def broadcast_einsum():
    return np.einsum('id,jd->ij', Q, K)   # sum over d for each (i,j) pair

t_loop, _       = benchmark(loop_dot,       repeats=1)   # very slow
t_numpy, r_np   = benchmark(numpy_dot)
t_bcast, r_bc   = benchmark(broadcast_einsum)

print_timing("Pure Python double loop",  t_loop)
print_timing("np.dot(Q, K.T)",           t_numpy,  baseline=t_loop)
print_timing("np.einsum('id,jd->ij')",   t_bcast,  baseline=t_loop)

print(f"\n  Results match: {np.allclose(r_np, r_bc)}")
print(f"  Score matrix shape: {r_np.shape}")
print("""
  np.dot(Q, K.T) and einsum give identical results.
  einsum notation ('id,jd->ij') is worth learning — PyTorch uses it
  extensively and it maps directly to the math in transformer papers.
  In the actual attention formula: scores = Q @ K.T / sqrt(D)
""")


# ─────────────────────────────────────────────────────────────
# OPERATION 4 — ReLU activation
# ─────────────────────────────────────────────────────────────
print_header("OPERATION 4 — ReLU Activation")
print("  Task: apply ReLU(x) = max(0, x) to every element")
print("  Array: 5,000,000 float values")
print("  ML use: activation function in every dense/conv layer\n")

SIZE = 5_000_000
x = np.random.randn(SIZE)

# Method 1 — Pure Python loop
def loop_relu():
    result = [0.0] * SIZE
    for i in range(SIZE):
        result[i] = x[i] if x[i] > 0 else 0.0
    return result

# Method 2 — NumPy vectorized with np.maximum
def numpy_relu():
    return np.maximum(0, x)

# Method 3 — Broadcasting / boolean mask (equivalent, slightly different style)
def broadcast_relu():
    result = x.copy()
    result[result < 0] = 0.0
    return result

t_loop, _       = benchmark(loop_relu,      repeats=1)
t_numpy, r_np   = benchmark(numpy_relu)
t_bcast, r_bc   = benchmark(broadcast_relu)

print_timing("Pure Python loop",        t_loop)
print_timing("np.maximum(0, x)",        t_numpy,  baseline=t_loop)
print_timing("Boolean mask (x<0)=0",   t_bcast,  baseline=t_loop)

loop_first5 = [max(0, float(v)) for v in x[:5]]
print(f"\n  Results match: {np.allclose(r_np[:5], loop_first5)}")
print("""
  np.maximum(0, x) is the canonical implementation and is what
  PyTorch calls under the hood for F.relu(). The boolean mask
  version is slightly slower because it makes two passes.
""")


# ─────────────────────────────────────────────────────────────
# OPERATION 5 — Softmax over a batch
# ─────────────────────────────────────────────────────────────
print_header("OPERATION 5 — Softmax over a Batch")
print("  Task: apply softmax to each row of a (batch × classes) matrix")
print("  Matrix: 2,000 samples × 1,000 class logits")
print("  ML use: final classifier layer, attention weights\n")

BATCH, CLASSES = 2_000, 1_000
logits = np.random.randn(BATCH, CLASSES)

# Method 1 — Pure Python loop over batch
def loop_softmax():
    result = np.empty_like(logits)
    for i in range(BATCH):
        row = logits[i]
        max_val = max(row)                        # numerical stability
        exps = [np.exp(v - max_val) for v in row]
        total = sum(exps)
        result[i] = [e / total for e in exps]
    return result

# Method 2 — NumPy vectorized
def numpy_softmax():
    shifted = logits - logits.max(axis=1, keepdims=True)   # stability
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)

# Method 3 — Same as above but using broadcasting more explicitly
def broadcast_softmax():
    # Subtract max per row for numerical stability
    shifted = logits - logits.max(axis=1)[:, np.newaxis]
    exps    = np.exp(shifted)
    sums    = exps.sum(axis=1)[:, np.newaxis]
    return exps / sums

t_loop, r_lp    = benchmark(loop_softmax,     repeats=1)
t_numpy, r_np   = benchmark(numpy_softmax)
t_bcast, r_bc   = benchmark(broadcast_softmax)

print_timing("Pure Python loop",        t_loop)
print_timing("NumPy vectorized",        t_numpy,  baseline=t_loop)
print_timing("Broadcasting explicit",   t_bcast,  baseline=t_loop)

print(f"\n  Results match (numpy vs broadcast): {np.allclose(r_np, r_bc)}")
print(f"  Each row sums to 1.0 (check first 3): {r_np[:3].sum(axis=1).round(8)}")
print("""
  Key insight: subtracting the row max before exp() is critical.
  Without it, np.exp(large_number) → inf and softmax breaks.
  Both the loop and NumPy versions do this — but NumPy does it
  in one vectorized pass instead of a Python loop.

  keepdims=True vs [:, np.newaxis] are interchangeable.
  keepdims is cleaner; np.newaxis is more explicit about what's happening.
""")


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print_header("SUMMARY — Why Vectorization Matters")
print("""
  Every operation above does the same computation.
  The only difference is HOW it's expressed.

  Loops:         Python interprets each iteration one by one.
  Vectorized:    NumPy drops into compiled C/Fortran — one call.
  Broadcasting:  Same compiled path, plus implicit shape expansion.

  For a single forward pass through a neural network,
  there are thousands of these operations in sequence.
  A 100x slowdown per op = training that takes days instead of minutes.

  This is why PyTorch, JAX, and TensorFlow all sit on top
  of the same idea: express operations on arrays, not loops.
""")
