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



