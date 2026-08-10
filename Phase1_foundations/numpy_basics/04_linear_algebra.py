import numpy as np
np.random.seed(0)

def print_header(title):
    print(f' {title}')

def check(label, val, expected, tol = 1e-6):
    ok = np.allclose(val, expected, atol=tol)
    print(f" { 'done' if ok else 'not_done'} {label}")
    if not ok:
        print(f' max error: {np.abs(val - expected).max():.2e}')

print_header("SECTION 1 — Matrix Inverse")

print("""
  ML connection:
    The normal equation for linear regression: w = (X.T @ X)^-1 @ X.T @ y
    Covariance matrix inversion appears in Gaussian processes and LDA.
    Understanding invertibility tells you when a weight matrix is degenerate.
""")

A = np.array([
    [4.0, 3.0, 2.0],
    [1.0, 5.0, 3.0],
    [2.0, 1.0, 6.0]
])

print(' Matrix A:\n',A)

#inversing a matrix via numpy

A_inv = np.linalg.inv(A)
print('\n A_inv (np.linalg.inv): \n', A_inv.round(6))

#verify
product= A @ A_inv
identity = np.eye(3)
check('A @ A_inv = I', product, identity, tol=1e-10)
print(' A @ A_inv: \n', product.round(10))

product2 = A_inv @ A
check('A_inv @ A = I', product2, identity, tol = 1e-10)

det = np.linalg.det(A)
print(f'\n det(A) = {det:.4f} (non-zero -> invertible)' )

# Determinant a singular matrix
A_singular = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
    ])

det_sing = np.linalg.det(A_singular)
print(f"\n  Singular matrix det = {det_sing:.2e}  (≈ 0 → NOT invertible)")
try:
    _= np.linalg.inv(A_singular)
    print('(NUmpy may not error but error result is numerical garbage)')
except np.linalg.LinAlgError as e:
    print(f' Error: {e}')


# Condition number - measures numerical stability of inversion

cond = np.linalg.cond(A)
print(f"\n  Condition number of A: {cond:.4f}")
print("""  Rule of thumb:
    cond < 100     → well-conditioned, inversion is safe
    cond > 1e6     → ill-conditioned, inversion will accumulate error
    cond → ∞       → singular (not invertible)
  In deep learning, ill-conditioned weight matrices lead to
  vanishing/exploding gradients — this is why initialisation matters.
""")

print_header('Section2 - PCA from scratch')

print("""
  ML connection:
    PCA is everywhere:
      - Dimensionality reduction before feeding data to a model
      - Visualising high-dimensional embeddings (reduce to 2D/3D)
      - The V matrix in SVD = principal components of the data
      - Understanding PCA from scratch = understanding what
        eigenvectors and eigenvalues actually mean in data terms
""") 

N_samples, D_feat = 200, 4
np.random.seed(7)

f1 = np.random.randn(N_samples)
f2 = np.random.randn(N_samples)
f3 = f1 + 0.1 * np.random.randn(N_samples)   # correlated with f1
f4 = np.random.randn(N_samples)
X_data = np.stack([f1, f2, f3, f4], axis=1)  # (200, 4)

print(f"  Data shape: {X_data.shape}")

# ── Step 1: Zero-mean (centre the data) ──────────────────────
X_mean    = X_data.mean(axis=0)               # (4,)
X_centred = X_data - X_mean                   # (200, 4)  broadcast
print(f"\n  Feature means before centring: {X_mean.round(3)}")
print(f"  Feature means after  centring: {X_centred.mean(axis=0).round(10)}")

# ── Step 2: Covariance matrix ─────────────────────────────────
# Cov = (1/(N-1)) * X_centred.T @ X_centred   → shape (D, D)
N = X_centred.shape[0]
Cov = (X_centred.T @ X_centred) / (N - 1)    # (4, 4)
print(f"\n  Covariance matrix:\n{Cov.round(3)}")
print("""  Notice the high Cov[0,2] and Cov[2,0] — that's features 1 and 3
  being correlated. PCA will collapse them into one component.""")

# ── Step 3: Eigendecomposition ────────────────────────────────
eigenvalues, eigenvectors = np.linalg.eigh(Cov)
# eigh is for symmetric matrices (covariance matrices always are)
# Returns eigenvalues in ascending order — we want descending
idx          = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]           # columns are eigenvectors

print(f"\n  Eigenvalues (variance explained per component):")
total_var = eigenvalues.sum()
for i, ev in enumerate(eigenvalues):
    print(f"    PC{i+1}: {ev:.4f}  ({100*ev/total_var:.1f}% of variance)")

# ── Step 4: Project data onto top-k components ────────────────
k = 2
W    = eigenvectors[:, :k]                    # (4, 2)  — projection matrix
X_2d = X_centred @ W                         # (200, 2) — reduced data
print(f"\n  Projected data shape: {X_2d.shape}")
print(f"  Reduced from {D_feat}D → {k}D while keeping "
      f"{100*eigenvalues[:k].sum()/total_var:.1f}% of variance")

# ── Verify against numpy's built-in SVD-based PCA ────────────
U, S, Vt = np.linalg.svd(X_centred, full_matrices=False)
X_2d_svd = X_centred @ Vt[:k].T
# Signs may differ (PCA components have arbitrary sign) — check abs
check("PCA projections match SVD (up to sign)",
      np.abs(X_2d), np.abs(X_2d_svd), tol=1e-8)

print("""
  The eigenvectors of the covariance matrix ARE the principal components.
  The eigenvalues tell you how much variance each component captures.
  In Transformers: the attention weight matrix's eigenspectrum tells you
  how many 'distinct' things the head is attending to.
""")


# ═══════════════════════════════════════════════════════════
# SECTION 3 — Solving Ax = b
# ═══════════════════════════════════════════════════════════
print_header("SECTION 3 — Solving Ax = b")

print("""
  ML connection:
    Linear regression normal equations:  (X.T @ X) w = X.T @ y
    Ridge regression:                    (X.T @ X + λI) w = X.T @ y
    Both are Ax = b problems.
    Understanding the two solution methods tells you when each is better.
""")

# Build a simple overdetermined system (more equations than unknowns)
# Simulate: 5 measurements, 3 unknown weights
A_sys = np.array([
    [2.0, 1.0, 3.0],
    [1.0, 4.0, 2.0],
    [3.0, 2.0, 1.0],
    [1.0, 1.0, 4.0],
    [2.0, 3.0, 1.0]
])
x_true = np.array([1.5, -0.5, 2.0])          # the "true" weights we want to find
b_vec  = A_sys @ x_true                       # perfect observations (no noise)

print(f"  A shape: {A_sys.shape}  |  b shape: {b_vec.shape}")
print(f"  True x: {x_true}")

# ── Method 1: np.linalg.solve  (only works for square A) ─────
A_sq = A_sys[:3, :]                           # take first 3 rows → 3×3
b_sq = b_vec[:3]
x_solve = np.linalg.solve(A_sq, b_sq)
print(f"\n  Method 1 — np.linalg.solve (square system):")
print(f"    x = {x_solve.round(6)}")
check("solve matches true x", x_solve, x_true)

# ── Method 2: Manual via inverse (A_inv @ b) ─────────────────
A_sq_inv = np.linalg.inv(A_sq)
x_manual = A_sq_inv @ b_sq
print(f"\n  Method 2 — Manual (A_inv @ b):")
print(f"    x = {x_manual.round(6)}")
check("manual inv matches true x", x_manual, x_true)
check("both methods agree", x_solve, x_manual)

# ── Method 3: Least squares for overdetermined system ────────
# This is the correct approach when A is not square (e.g. regression)
x_lstsq, residuals, rank, sv = np.linalg.lstsq(A_sys, b_vec, rcond=None)
print(f"\n  Method 3 — np.linalg.lstsq (overdetermined, {A_sys.shape[0]} eqs):")
print(f"    x = {x_lstsq.round(6)}")
print(f"    rank = {rank}  |  singular values = {sv.round(4)}")
check("lstsq matches true x (no noise)", x_lstsq, x_true)

# Add noise to b and show least squares is robust
noise   = 0.05 * np.random.randn(5)
b_noisy = b_vec + noise
x_noisy, _, _, _ = np.linalg.lstsq(A_sys, b_noisy, rcond=None)
print(f"\n  With noise: x_lstsq = {x_noisy.round(4)}  (should be close to {x_true})")

print("""
  When to use each:
    np.linalg.solve  — fast, exact, only for square well-conditioned A
    A_inv @ b        — never in production (slower + less numerically stable)
    np.linalg.lstsq  — the right tool for regression and overdetermined systems

  In practice: PyTorch's nn.Linear uses highly optimised BLAS routines
  for the forward pass (A @ x + b), not linalg.solve.
  But the normal equations for analytical linear regression = lstsq.
""")


# ═══════════════════════════════════════════════════════════
# SECTION 4 — SVD Basics
# ═══════════════════════════════════════════════════════════
print_header("SECTION 4 — SVD (Singular Value Decomposition)")

print("""
  ML connection:
    SVD is the backbone of:
      - PCA (as shown above — S² = eigenvalues of covariance matrix)
      - Low-rank approximation (compress weight matrices)
      - LoRA fine-tuning of LLMs (add low-rank update ΔW = A @ B)
      - Recommendation systems (matrix factorisation)
      - Understanding the 'intrinsic rank' of an attention layer
""")

# Decompose a 5×4 data matrix
M_svd = np.random.randn(5, 4)
U, S, Vt = np.linalg.svd(M_svd, full_matrices=False)
# U: (5, 4)  S: (4,)  Vt: (4, 4)
# M = U @ np.diag(S) @ Vt

print(f"  M shape: {M_svd.shape}")
print(f"  U shape: {U.shape}  (left singular vectors)")
print(f"  S shape: {S.shape}  (singular values)")
print(f"  Vt shape: {Vt.shape}  (right singular vectors, transposed)")
print(f"\n  Singular values: {S.round(4)}")

# Verify reconstruction
M_reconstructed = U @ np.diag(S) @ Vt
check("U @ diag(S) @ Vt = M", M_reconstructed, M_svd)

# Low-rank approximation — keep only top-k singular values
print("\n  Low-rank approximation (keep top k singular values):")
for k in [1, 2, 3, 4]:
    M_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    error = np.linalg.norm(M_svd - M_k, 'fro')
    var_explained = (S[:k]**2).sum() / (S**2).sum() * 100
    print(f"    k={k}: Frobenius error = {error:.4f}  |  "
          f"variance explained = {var_explained:.1f}%")

print("""
  Key insight: if 90% of the variance is in the top 2 singular values,
  a rank-2 approximation captures the essential structure of the matrix
  at a fraction of the storage cost.

  This is exactly what LoRA does to fine-tune LLMs cheaply:
  instead of updating the full weight matrix W (huge),
  it learns two small matrices A and B where ΔW = A @ B (low-rank).
""")

