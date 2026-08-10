import numpy as np
import time

np.random.seed(42)


def print_header(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)
class CountUp:
    def __init__(self, start, stop):
        self.current = start
        self.stop    = stop

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.stop:
            raise StopIteration
        val = self.current
        self.current += 1
        return val

counter = CountUp(0, 5)
result  = list(counter)
print(f"  CountUp(0, 5): {result}")
print(f"  Used as for loop: ", end="")
for v in CountUp(0, 5):
    print(v, end=" ")
print()

print_header("SimpleDataLoader")

class SimpleDataLoader:
    """
    Batches and optionally shuffles a NumPy dataset.

    Usage:
        loader = SimpleDataLoader(X, y, batch_size=32, shuffle=True)
        for X_batch, y_batch in loader:
            ...  # X_batch: (batch_size, features)

    Args:
        X          — feature array of shape (N, ...)
        y          — label array of shape  (N, ...)
        batch_size — number of samples per batch
        shuffle    — randomise order each epoch
        drop_last  — discard final batch if smaller than batch_size
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        assert len(X) == len(y), "X and y must have the same number of samples"
        self.X          = X
        self.y          = y
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.drop_last  = drop_last
        self.n_samples  = len(X)

        # Compute number of batches
        full_batches    = self.n_samples // batch_size
        has_remainder   = (self.n_samples % batch_size) > 0
        self.n_batches  = full_batches if drop_last else full_batches + has_remainder

    # ── Iterator protocol ────────────────────────────────────
    def __iter__(self):
        """Called at the start of each for loop — resets and shuffles."""
        if self.shuffle:
            self._indices = np.random.permutation(self.n_samples)
        else:
            self._indices = np.arange(self.n_samples)
        self._cursor = 0
        return self

    def __next__(self):
        """Return next (X_batch, y_batch) or raise StopIteration."""
        if self._cursor >= self.n_samples:
            raise StopIteration

        end = self._cursor + self.batch_size

        # drop_last: if the remaining samples are fewer than batch_size, stop
        if self.drop_last and (self.n_samples - self._cursor) < self.batch_size:
            raise StopIteration

        batch_idx     = self._indices[self._cursor:end]
        self._cursor  = end
        return self.X[batch_idx], self.y[batch_idx]

    # ── Convenience ──────────────────────────────────────────
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.n_batches

    def __repr__(self):
        return (f"SimpleDataLoader("
                f"n_samples={self.n_samples}, "
                f"batch_size={self.batch_size}, "
                f"n_batches={self.n_batches}, "
                f"shuffle={self.shuffle})")


# ── Smoke test ────────────────────────────────────────────────
N, D, C = 100, 8, 3    # 100 samples, 8 features, 3 classes
X_demo  = np.random.randn(N, D)
y_demo  = np.random.randint(0, C, size=N)

loader = SimpleDataLoader(X_demo, y_demo, batch_size=16, shuffle=True)
print(f"\n  {loader}")
print(f"  len(loader) = {len(loader)} batches")

batches = list(loader)
print(f"\n  Iterating once:")
for i, (xb, yb) in enumerate(loader):
    print(f"    batch {i}: X={xb.shape}  y={yb.shape}")


# ═══════════════════════════════════════════════════════════
# SHUFFLE VERIFICATION
# ═══════════════════════════════════════════════════════════
print_header("Shuffle Verification")

print("""
  Key property: shuffling happens fresh at the start of EACH epoch.
  Same data, different order every time — this is what prevents
  the model from memorising the order of training examples.
""")

small_X = np.arange(10).reshape(10, 1).astype(float)
small_y = np.arange(10)
small_loader = SimpleDataLoader(small_X, small_y, batch_size=10, shuffle=True)

print("\n  10 samples, batch_size=10 (one big batch), 3 epochs:")
for epoch in range(3):
    for xb, yb in small_loader:
        print(f"    Epoch {epoch+1}: {yb.tolist()}")

no_shuffle = SimpleDataLoader(small_X, small_y, batch_size=10, shuffle=False)
print("\n  Same loader with shuffle=False:")
for epoch in range(2):
    for xb, yb in no_shuffle:
        print(f"    Epoch {epoch+1}: {yb.tolist()}")


print_header("drop_last=True vs False")

print("""
  If N=105 and batch_size=32:
    drop_last=False: batches of [32, 32, 32, 9]   (last is smaller)
    drop_last=True:  batches of [32, 32, 32]       (last 9 discarded)

  Use drop_last=True when:
    - BatchNorm statistics need consistent batch sizes
    - You're computing stats that assume equal-sized batches
""")

N_drop = 105
X_drop = np.random.randn(N_drop, 4)
y_drop = np.zeros(N_drop)

loader_keep = SimpleDataLoader(X_drop, y_drop, batch_size=32,
                                shuffle=False, drop_last=False)
loader_drop = SimpleDataLoader(X_drop, y_drop, batch_size=32,
                                shuffle=False, drop_last=True)

print(f"\n  drop_last=False: {[len(xb) for xb, _ in loader_keep]} "
      f"→ {len(loader_keep)} batches")
print(f"  drop_last=True:  {[len(xb) for xb, _ in loader_drop]} "
      f"→ {len(loader_drop)} batches")


print_header("DataLoader in a Real Training Loop")

print("""
  This is the exact pattern every PyTorch training loop uses.
  We swap in our SimpleDataLoader — the loop body stays identical.
""")

# Inline minimal model and loss for demo
class TinyLinear:
    def __init__(self, in_f, out_f):
        self.W = np.random.randn(out_f, in_f) * 0.1
        self.b = np.zeros(out_f)
    def __call__(self, x):
        return x @ self.W.T + self.b

def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

# Synthetic regression: y = 2*x + 1 (learn from data)
N_train = 200
X_train = np.random.randn(N_train, 1)
y_train = 2.0 * X_train + 1.0 + 0.1 * np.random.randn(N_train, 1)

train_loader = SimpleDataLoader(X_train, y_train,
                                 batch_size=32, shuffle=True)
model        = TinyLinear(1, 1)
lr           = 0.01
n_epochs     = 10

print(f"\n  Task: learn y = 2x + 1 from {N_train} samples")
print(f"  Loader: {train_loader}")
print(f"\n  Training for {n_epochs} epochs:")

for epoch in range(n_epochs):
    epoch_loss = 0.0
    n_batches  = 0

    for X_batch, y_batch in train_loader:
        # Forward
        pred  = model(X_batch)
        loss  = mse_loss(pred, y_batch)

        # Backward (manual gradient for linear layer)
        err   = pred - y_batch                         # (batch, 1)
        dW    = (2 / len(X_batch)) * (err.T @ X_batch) # (1, 1)
        db    = (2 / len(X_batch)) * err.sum(axis=0)   # (1,)

        # Update
        model.W -= lr * dW
        model.b -= lr * db

        epoch_loss += loss
        n_batches  += 1

    if epoch % 2 == 0 or epoch == n_epochs - 1:
        print(f"    Epoch {epoch+1:>2}/{n_epochs}  "
              f"loss={epoch_loss/n_batches:.6f}  "
              f"W={model.W[0,0]:.4f}  b={model.b[0]:.4f}")

print(f"\n  Target: W=2.0, b=1.0")
print(f"  Learned: W={model.W[0,0]:.4f}, b={model.b[0]:.4f}  ✓")
