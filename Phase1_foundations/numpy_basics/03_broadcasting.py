import numpy as np

np.random.seed(42)   # reproducible outputs

def print_header(title):
    print("\n" + "=" * 58)
    print(f"  {title}")
    print("=" * 58)

def check(name, yours, reference, tol=1e-6):
    ok = np.allclose(yours, reference, atol=tol)
    status = "PASS" if ok else "FAIL"
    print(f"\n  {status}  {name}")
    if not ok:
        diff = np.abs(yours - reference).max()
        print(f"Max absolute error: {diff:.2e}")
    return ok

print_header("EXERCISE 1 — Batch Matrix Multiply (no np.matmul)")

print("""
  Context:
    In a multi-head attention layer, we compute Q @ K.T independently
    for each attention head and each sample in the batch — all in one call.
    That operation is a batched matrix multiply over shape (B, H, T, T).

  Your task:
    Given A of shape (B, M, K) and B of shape (B, K, N),
    produce C of shape (B, M, N) where C[b] = A[b] @ B[b]
    for every batch index b.

    Rules: no np.matmul, no np.dot, no np.tensordot, no np.einsum.
    Use only: *, +, np.sum, np.expand_dims / reshape, and slicing.
""")

B_size, M, K, N = 8, 6, 10, 4
A = np.random.randn(B_size, M, K)
B = np.random.randn(B_size, K, N)

def bmm_reference(A, B):
    return np.matmul(A, B)   # shape: (B, M, N)

def bmm_scratch(A, B):
    A_exp = A[:, :, :, np.newaxis]    # (B, M, K, 1)
    B_exp = B[:, np.newaxis, :, :]    # (B, 1, K, N)
    products = A_exp * B_exp          # (B, M, K, N)  — broadcast over M and N
    return products.sum(axis=2)       # (B, M, N)     — sum over K


C_ref   = bmm_reference(A, B)
C_yours = bmm_scratch(A, B)

print(f"  A shape: {A.shape}  x  B shape: {B.shape}  ->  C shape: {C_ref.shape}")
print(f"\n  Reference C[0]:\n{C_ref[0].round(3)}")
print(f"\n  Your C[0]:\n{C_yours[0].round(3)}")
check("bmm_scratch vs np.matmul", C_yours, C_ref)

print("""
  Why expand_dims then sum?
    A_exp * B_exp creates the full outer-product tensor over M, K, N.
    Summing over the K axis collapses the contraction — that IS
    matrix multiplication, expressed as broadcast-multiply-then-sum.

  This is also exactly what np.einsum('bik,bkj->bij', A, B) does
  internally. Knowing the expand-and-sum form means you can read
  einsum notation and know what's actually happening in memory.
""")

print_header("EXERCISE 2 — Batch L2 Normalisation (no np.linalg.norm)")

print("""
  Context:
    Normalised vectors appear everywhere in Transformers:
      - Query and key vectors are often L2-normalised before dot product
        (this is the core idea behind Cosine Attention)
      - Word embeddings are sometimes unit-normalised before lookup
      - The SWAP test in Task I works with normalised quantum states

  Your task:
    Given X of shape (N, D) — N vectors, each of dimension D —
    return X_norm of shape (N, D) where every row has L2 norm = 1.0.

    L2 norm of a vector v: ||v|| = sqrt(v[0]^2 + v[1]^2 + ... + v[D-1]^2)
    Normalised vector:      v_hat = v / ||v||

    Rules: no np.linalg.norm, no sklearn.preprocessing.normalize.
    Use only: **, *, /, np.sqrt, np.sum, keepdims.
""")

N_vecs, D_dim = 1000, 128
X = np.random.randn(N_vecs, D_dim)

X[5] = X[5] * 1e-10   # near-zero — dividing by its norm is risky

def l2_norm_reference(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-8)   # epsilon for stability

def l2_norm_scratch(X):
    sq_sums = (X ** 2).sum(axis=1, keepdims=True)   # (N, 1)
    norms   = np.sqrt(sq_sums) + 1e-8               # (N, 1)
    return X / norms                                 # (N, D)

X_ref = l2_norm_reference(X)
X_yours = l2_norm_scratch(X)

check("l2_norm_scratch vs np.linalg.norm", X_yours, X_ref)

row_norms = np.sqrt((X_yours ** 2).sum(axis=1))
print(f"\n  Row norms — min: {row_norms.min():.6f}  "
      f"max: {row_norms.max():.6f}  "
      f"mean: {row_norms.mean():.6f}")
print(f"  (all should be ≈ 1.0, near-zero row is handled by epsilon)")

v = X[0]
v_norm = X_yours[0]
print(f"\n  Sample vector X[0]:norm = {np.sqrt((v**2).sum()):.4f}")
print(f"  Normalised  X_norm[0]:norm = {np.sqrt((v_norm**2).sum()):.6f}")

print("""
  The epsilon (+1e-8) is a numerical stability trick you'll see
  in every production implementation of normalisation.
  PyTorch's F.normalize() uses the same pattern internally.

  Cosine similarity between two L2-normalised vectors is just
  their dot product — so after normalising, a matmul gives you
  a full matrix of cosine similarities in one operation.
""")

print_header("EXERCISE 3 — Pairwise Euclidean Distance (no scipy)")

print("""
  Context:
    Distance matrices appear in:
      - k-NN classification (find the N nearest neighbours)
      - Attention with RBF kernels (exp(-||qi - kj||^2 / sigma))
      - Graph construction for GNNs (Task II): edges connect
        particles whose feature vectors are within distance r
      - Clustering algorithms (k-means assignment step)

  Your task:
    Given A of shape (M, D) and B of shape (N, D),
    return D_mat of shape (M, N) where:
        D_mat[i, j] = sqrt( sum_d (A[i,d] - B[j,d])^2 )

    Brute-force (triple loop) is easy but O(M*N*D).
    The fast version uses the identity:
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a·b

    Rules: no scipy, no sklearn, no cdist.
    Use only: **, np.sqrt, np.sum, np.dot / matmul, broadcasting.
""")

M_pts, N_pts, D_pts = 50, 80, 32
A_pts = np.random.randn(M_pts, D_pts)
B_pts = np.random.randn(N_pts, D_pts)

def dist_reference(A, B):
    M, D = A.shape
    N    = B.shape[0]
    D_mat = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            diff = A[i] - B[j]
            D_mat[i, j] = np.sqrt((diff ** 2).sum())
    return D_mat

def dist_scratch(A, B):
    # Step 1: squared norms
    sq_A = (A ** 2).sum(axis=1, keepdims=True)   # (M, 1)
    sq_B = (B ** 2).sum(axis=1, keepdims=True).T  # (1, N)

    # Step 2: pairwise dot products
    dot = A @ B.T                                 # (M, N)
    # Step 3: combine — clip at 0 to avoid sqrt of tiny negatives from floats
    sq_dist = sq_A + sq_B - 2 * dot              # (M, N)
    sq_dist = np.clip(sq_dist, 0, None)           # numerical safety

    return np.sqrt(sq_dist)                       # (M, N)


D_ref   = dist_reference(A_pts, B_pts)
D_yours = dist_scratch(A_pts, B_pts)

check("dist_scratch vs brute-force loop", D_yours, D_ref, tol=1e-5)

print(f"\n  Output shape: {D_yours.shape}  (M={M_pts} points × N={N_pts} points)")
print(f"  Min distance:  {D_yours.min():.4f}")
print(f"  Max distance:  {D_yours.max():.4f}")
print(f"  Mean distance: {D_yours.mean():.4f}")

# Bonus: find nearest neighbour in B for each point in A
nearest_idx  = D_yours.argmin(axis=1)     # (M,) — index of closest B for each A
nearest_dist = D_yours.min(axis=1)        # (M,) — distance to that B
print(f"\n  Nearest neighbour indices (first 5): {nearest_idx[:5]}")
print(f"  Nearest neighbour distances (first 5): {nearest_dist[:5].round(4)}")

# Connection to GNN graph construction
threshold = np.percentile(D_yours, 20)    # connect if in closest 20%
adj_matrix = (D_yours < threshold).astype(int)
print(f"\n  Adjacency matrix (threshold={threshold:.2f}):")
print(f"    shape: {adj_matrix.shape}")
print(f"    edges: {adj_matrix.sum()}  (out of {M_pts*N_pts} possible)")
print(f"    This IS the graph construction step for Task II GNN")

print("""
  The ||a||^2 + ||b||^2 - 2*a·b identity reduces the problem to
  one matrix multiply + two norm vectors. This is the same trick
  used by FAISS (Facebook's fast similarity search library) and
  by GPU-accelerated k-NN in cuML.

  For Task II: you'll build the graph by computing this distance
  matrix over particle feature vectors, then adding an edge
  between particles i and j if dist[i,j] < some cutoff radius.
""")
print_header("BONUS — Combine all 3 into a Mini Attention Step")

print("""
  Scaled dot-product attention in a Transformer:
      1. Normalise Q and K (Exercise 2)
      2. Compute Q @ K.T  (Exercise 1, single-batch case)
      3. Scale by 1/sqrt(D)
      4. Softmax → attention weights

  Let's chain exercises 1-2 together and compute attention weights.
""")

T_seq, D_attn = 16, 32   # sequence length, head dimension
Q_raw = np.random.randn(T_seq, D_attn)
K_raw = np.random.randn(T_seq, D_attn)
V_raw = np.random.randn(T_seq, D_attn)

Q_n = l2_norm_scratch(Q_raw)
K_n = l2_norm_scratch(K_raw)

scores = Q_n @ K_n.T                      # (T, T)

scores = scores / np.sqrt(D_attn)

scores = scores - scores.max(axis=1, keepdims=True)
attn_weights = np.exp(scores)
attn_weights = attn_weights / attn_weights.sum(axis=1, keepdims=True)

output = attn_weights @ V_raw             # (T, D)

print(f"  Q shape: {Q_n.shape}")
print(f"  K shape: {K_n.shape}")
print(f"  Attention weights shape: {attn_weights.shape}")
print(f"  Output shape: {output.shape}")
print(f"\n  Attention weights (first row, should sum to 1.0):")
print(f"    {attn_weights[0].round(4)}")
print(f"    sum = {attn_weights[0].sum():.8f}")
print(f"\n  Output (first token, first 8 dims):")
print(f"    {output[0, :8].round(4)}")

print("""
  That's it. That's scaled dot-product attention.
  In Phase 3 you'll wrap this in a class, add learned W_Q/W_K/W_V
  projections, stack multiple heads, and you'll have a full
  multi-head attention layer.

  Everything you just built in these 3 exercises is the math inside it.
""")

print_header("DONE — 03_broadcasting.py complete")

