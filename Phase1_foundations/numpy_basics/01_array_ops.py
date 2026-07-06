import numpy as np

print("\n" + "="*50)
print("SECTION 1 — Array Creation")
print("="*50)

zeros = np.zeros((3, 4))          # 3 rows, 4 columns of 0.0
ones  = np.ones((2, 3))           # 2 rows, 3 columns of 1.0
print("np.zeros((3,4)):\n", zeros)
print("\nnp.ones((2,3)):\n", ones)

rand_normal = np.random.randn(3, 3)   # Standard normal (mean=0, std=1)
rand_uniform = np.random.rand(3, 3)   # Uniform [0, 1)
print("\nnp.random.randn(3,3):\n", rand_normal)
print("\nnp.random.rand(3,3):\n", rand_uniform)

arange  = np.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)       # 5 evenly spaced points from 0 to 1
print("\nnp.arange(0, 10, 2):", arange)
print("np.linspace(0, 1, 5):", linspace)

identity = np.eye(4)
print("\nnp.eye(4):\n", identity)

filled = np.full((2, 5), 7.0)
print("\nnp.full((2,5), 7.0):\n", filled)

arr = np.random.randn(4, 3)
print("\nArray shape:", arr.shape)       # (4, 3)
print("Array dtype:", arr.dtype)         # float64
print("Array ndim:", arr.ndim)           # 2
print("Array size:", arr.size) 

print("\n" + "="*50)
print("SECTION 2 — Indexing and Slicing")
print("="*50)

a = np.array([10, 20, 30, 40, 50])
print("\n1D array:", a)
print("a[0]       =", a[0])        # First element: 10
print("a[-1]      =", a[-1])       # Last element: 50
print("a[1:4]     =", a[1:4])      # [20, 30, 40]
print("a[::2]     =", a[::2])      # Every other: [10, 30, 50]
print("a[::-1]    =", a[::-1])  

B = np.array([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12]
])
print("\n2D array B:\n", B)
print("B[0, :]      =", B[0, :])       # First row: [1 2 3 4]
print("B[:, 1]      =", B[:, 1])       # Second column: [2 6 10]
print("B[1:, 2:]    =\n", B[1:, 2:])  # Bottom-right 2x2 submatrix
print("B[0, -1]     =", B[0, -1])  

mask = B > 6
print("\nMask (B > 6):\n", mask)
print("B[B > 6]     =", B[mask])  

rows = np.array([0, 2])
print("B[[0,2], :]  =\n", B[rows, :])

C = np.random.randint(0, 255, size=(4, 6, 6))   # 4 images, 6x6 pixels
print("\n3D array C shape:", C.shape)
print("First image (C[0]):\n", C[0])
print("Pixel at batch=1, row=2, col=3:", C[1, 2, 3])
print("All top-left pixels: C[:, 0, 0] =", C[:, 0, 0])

print("\n" + "="*50)
print("SECTION 3 — Reshaping")
print("="*50)

x = np.arange(24)          # [0, 1, 2, ..., 23]
print("\nOriginal x:", x, "| shape:", x.shape)

x_2d = x.reshape(4, 6)
print("\nreshape(4, 6):\n", x_2d)

x_3d = x.reshape(2, 3, 4)
print("\nreshape(2, 3, 4):\n", x_3d)

x_auto = x.reshape(6, -1)  # -1 → 24/6 = 4
print("\nreshape(6, -1) → shape:", x_auto.shape)

flat = x_3d.flatten()
print("\nflatten() → shape:", flat.shape, "| first 5:", flat[:5])

raveled = x_3d.ravel()
print("ravel()   → shape:", raveled.shape)

squeezable = np.zeros((1, 4, 1, 3))
print("\nBefore squeeze:", squeezable.shape)
print("After squeeze: ", np.squeeze(squeezable).shape)    # (4, 3)


single_image = np.random.randn(28, 28)               # One MNIST image
batched = np.expand_dims(single_image, axis=0)       # Add batch dim
print("\nImage shape:", single_image.shape)
print("After expand_dims(axis=0):", batched.shape)   # (1, 28, 28)


matrix = np.random.randn(3, 5)
print("\nmatrix shape:", matrix.shape)
print("matrix.T shape:", matrix.T.shape)             # (5, 3)

arr_3d = np.random.randn(2, 3, 4)
transposed = arr_3d.transpose(0, 2, 1)              # Swap last two axes
print("3D transpose(0,2,1):", arr_3d.shape, "→", transposed.shape)

print("\n" + "="*50)
print("SECTION 4 — Axis Operations")
print("="*50)

M = np.array([
    [2.0, 3.0, 1.0, 5.0, 4.0],
    [1.0, 4.0, 2.0, 3.0, 5.0],
    [5.0, 1.0, 4.0, 2.0, 3.0],
    [3.0, 5.0, 3.0, 1.0, 2.0]
])
print("\nData matrix M (4 samples × 5 features):\n", M)

print("\nSum along axis=0 (per feature):", M.sum(axis=0))
print("Sum along axis=1 (per sample): ", M.sum(axis=1))

print("\nMean along axis=0:", M.mean(axis=0))
print("Mean along axis=1:", M.mean(axis=1))

print("\nMax  along axis=0:", M.max(axis=0))
print("Max  along axis=1:", M.max(axis=1))

print("\nArgmax along axis=1 (index of top feature per sample):", M.argmax(axis=1))

col_means = M.mean(axis=0, keepdims=True)   # shape (1, 5) not (5,)
print("\nColumn means with keepdims=True:", col_means.shape)

M_normalised = M - col_means
print("Normalised M (first row should sum ≈ 0):", M_normalised.mean(axis=0).round(10))

# Cumulative ops
print("\nCumulative sum of first row:", np.cumsum(M[0]))

print("\n" + "="*50)
print("SECTION 5 — Broadcasting")
print("="*50)
print("""
Broadcasting rule:
  Dimensions are compared from the right.
  Two dims are compatible if they are equal OR one of them is 1.
  The 1-dim is 'stretched' to match the other.
""")

print("── Example 1: Add bias to every row ──")
X = np.ones((5, 4))          # 5 samples, 4 features  → shape (5, 4)
b = np.array([1, 2, 3, 4])   # Bias vector            → shape (4,)
# Broadcasting: (5, 4) + (4,) → (4,) is treated as (1, 4) → broadcast to (5, 4)
result = X + b
print("X shape:", X.shape, "+ b shape:", b.shape, "→ result shape:", result.shape)
print("First row:", result[0])   # [2. 3. 4. 5.]
print("Last row: ", result[-1])

try:
    wrong_b = np.array([1, 2, 3])  # shape (3,) — incompatible with (5, 4)
    _ = X + wrong_b
except ValueError as e:
    print("\n⚠ Expected error without broadcasting:", e)

print("\n── Example 2: Feature normalisation (like LayerNorm) ──")
batch = np.random.randn(8, 16)                        # 8 samples, 16 features
mean = batch.mean(axis=1, keepdims=True)              # shape (8, 1)
std  = batch.std(axis=1, keepdims=True) + 1e-8        # shape (8, 1), eps avoids /0
# Broadcasting: (8, 16) - (8, 1) → (8, 1) stretches across 16 features
normalised = (batch - mean) / std                     # shape (8, 16)
print("batch shape:     ", batch.shape)
print("mean shape:      ", mean.shape)
print("normalised shape:", normalised.shape)
print("Normalised mean per sample (should be ≈ 0):", normalised.mean(axis=1).round(8))
print("Normalised std  per sample (should be ≈ 1):", normalised.std(axis=1).round(8))

print("\n── Example 3: Pairwise squared distances ──")
A = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])   # 3 points in 2D  → shape (3, 2)

B_pts = np.array([[1.0, 1.0],
                  [4.0, 4.0]])  # 2 query points → shape (2, 2)

diff = A[:, np.newaxis, :] - B_pts[np.newaxis, :, :]   # (3, 2, 2)
sq_dist = (diff ** 2).sum(axis=-1)                      # (3, 2)
print("A shape:", A.shape, " | B shape:", B_pts.shape)
print("Pairwise squared distances (3 points × 2 queries):\n", sq_dist)
print("(Row i, col j) = squared distance from A[i] to B[j]")

# Verify one entry manually: dist(A[0], B[0]) = (1-1)^2 + (2-1)^2 = 1
manual = (1-1)**2 + (2-1)**2
print("\nManual check dist(A[0], B[0]):", manual, "| Array gives:", sq_dist[0, 0])

print("\n" + "="*50)
print("DONE — 01_array_ops.py complete")
print("="*50)
print("""
What to do next:
  1. Re-read any section that felt unclear and experiment by changing values
  2. Open 02_vectorization.py and time loop vs vectorized vs broadcasting
  3. Commit this file:
     git add phase1_foundations/numpy_basics/01_array_ops.py
     git commit -m "feat: numpy array ops — creation, slicing, reshape, axes, broadcasting"
""")