import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — saves to file instead of showing window
import matplotlib.pyplot as plt
import os

np.random.seed(0)


def print_header(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)

# Base XOR examples
X_xor = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=float)
y_xor = np.array([[0], [1], [1], [0]], dtype=float)

# Expand with noise to make training more interesting
np.random.seed(1)
N_copies = 200
noise    = 0.1
X_train  = np.tile(X_xor, (N_copies // 4, 1)) + noise * np.random.randn(N_copies, 2)
y_train  = np.tile(y_xor, (N_copies // 4, 1))

# Shuffle
perm    = np.random.permutation(N_copies)
X_train = X_train[perm]
y_train = y_train[perm]

print(f"  Training set: {X_train.shape}  labels: {y_train.shape}")
print(f"  Class balance: {int(y_train.sum())} positives / "
      f"{int(len(y_train) - y_train.sum())} negatives")

print_header("STEP 2 — Initialise Parameters")

print("""
  We store all parameters in a plain dict.
  No classes — just arrays.  This is intentional:
  it forces you to track every gradient manually,
  which is exactly what autograd does for you in PyTorch.
""")

def init_params(n_in=2, n_hidden=4, n_out=1, seed=42):
    """
    Xavier (Glorot) initialisation for Tanh networks.
    scale = sqrt(1 / n_in)
    """
    np.random.seed(seed)
    scale1 = np.sqrt(1.0 / n_in)
    scale2 = np.sqrt(1.0 / n_hidden)
    return {
        'W1': np.random.randn(n_hidden, n_in)  * scale1,   # (4, 2)
        'b1': np.zeros((1, n_hidden)),                       # (1, 4)
        'W2': np.random.randn(n_out, n_hidden) * scale2,   # (1, 4)
        'b2': np.zeros((1, n_out)),                          # (1, 1)
    }

params = init_params()
for name, arr in params.items():
    print(f"  {name}: shape={arr.shape}  "
          f"mean={arr.mean():.4f}  std={arr.std():.4f}")

print_header("STEP 3 — Forward Pass")

def sigmoid(z):
    """σ(z) = 1 / (1 + e^-z)"""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def tanh_act(z):
    """tanh(z)"""
    return np.tanh(z)

def forward(X, params):
    """
    Two-layer network forward pass.

    Layer 1: z1 = X @ W1.T + b1   →  a1 = tanh(z1)
    Layer 2: z2 = a1 @ W2.T + b2  →  a2 = sigmoid(z2)

    Returns:
        a2    — output probabilities, shape (batch, 1)
        cache — all intermediate values needed for backprop
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    z1 = X @ W1.T + b1       # (N, 4)
    a1 = tanh_act(z1)         # (N, 4)

    z2 = a1 @ W2.T + b2      # (N, 1)
    a2 = sigmoid(z2)          # (N, 1)

    cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache

# Smoke test
a2_test, _ = forward(X_xor, params)
print(f"  Forward pass on XOR base (4 examples):")
print(f"  Output: {a2_test.T[0].round(4)}  (random init, not trained yet)")


print("""
  BCE(y, ŷ) = -1/N * Σ [ y * log(ŷ) + (1-y) * log(1-ŷ) ]

  The epsilon (1e-15) prevents log(0) = -inf.
  This is the standard loss for binary classification.
""")

def bce_loss(y_true, y_pred):
    """Binary cross-entropy. y_pred clipped to avoid log(0)."""
    eps     = 1e-15
    y_pred  = np.clip(y_pred, eps, 1 - eps)
    N       = len(y_true)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

loss_init = bce_loss(y_xor, a2_test)
print(f"  Initial loss (random init): {loss_init:.4f}")
print(f"  (Random binary classifier baseline ≈ log(2) = {np.log(2):.4f})")


print("""
  We apply the chain rule layer by layer, right to left.

  dL/da2 = -(y/a2 - (1-y)/(1-a2))           ← BCE derivative
  dL/dz2 = a2 - y                             ← combined sigmoid + BCE
           (this simplification is why sigmoid + BCE is standard)

  dL/dW2 = a1.T @ dz2  (averaged over batch)
  dL/db2 = dz2.mean(axis=0)

  dL/da1 = dz2 @ W2

  dL/dz1 = da1 * (1 - a1^2)                  ← tanh derivative
           tanh'(z) = 1 - tanh(z)^2

  dL/dW1 = X.T @ dz1  (averaged over batch)
  dL/db1 = dz1.mean(axis=0)
""")

def backward(y_true, params, cache):
    """
    Compute gradients of BCE loss w.r.t. all parameters.

    Returns: dict of gradients with same keys as params.
    """
    W2     = params['W2']
    a1, a2 = cache['a1'], cache['a2']
    X      = cache['X']
    N      = len(y_true)

    # ── Output layer ──────────────────────────────────────
    # Combined sigmoid + BCE gradient: dL/dz2 = a2 - y
    dz2 = (a2 - y_true) / N          # (N, 1)

    dW2 = dz2.T @ a1                  # (1, 4)
    db2 = dz2.sum(axis=0, keepdims=True)  # (1, 1)

    # ── Hidden layer ──────────────────────────────────────
    da1 = dz2 @ W2                    # (N, 4)  — backprop through W2

    # Tanh derivative: d/dz tanh(z) = 1 - tanh(z)^2 = 1 - a1^2
    dz1 = da1 * (1 - a1 ** 2)        # (N, 4)

    dW1 = dz1.T @ X                   # (4, 2)
    db1 = dz1.sum(axis=0, keepdims=True)  # (1, 4)

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}



def numerical_grad(params, param_key, i, j, X, y, eps=1e-5):
    """Finite difference gradient for params[param_key][i, j]."""
    p = params[param_key]

    p[i, j] += eps
    loss_plus, _ = forward(X, params)
    loss_plus     = bce_loss(y, loss_plus)

    p[i, j] -= 2 * eps
    loss_minus, _ = forward(X, params)
    loss_minus     = bce_loss(y, loss_minus)

    p[i, j] += eps   # restore
    return (loss_plus - loss_minus) / (2 * eps)

# Use a small subset so the check is fast
X_chk = X_xor
y_chk = y_xor

a2_chk, cache_chk = forward(X_chk, params)
grads_analytic     = backward(y_chk, params, cache_chk)

print(f"\n  Checking W1 (4 entries) and W2 (4 entries):\n")
all_ok = True
for key, (r, c) in [('W1', (0, 0)), ('W1', (1, 1)),
                     ('W1', (2, 0)), ('W1', (3, 1)),
                     ('W2', (0, 0)), ('W2', (0, 1)),
                     ('W2', (0, 2)), ('W2', (0, 3))]:
    numeric  = numerical_grad(params, key, r, c, X_chk, y_chk)
    analytic = grads_analytic[key][r, c]
    rel_err  = abs(numeric - analytic) / (abs(numeric) + abs(analytic) + 1e-15)
    ok       = rel_err < 1e-4
    all_ok   = all_ok and ok
    print(f"  {key}[{r},{c}]:  analytic={analytic:+.6f}  "
          f"numeric={numeric:+.6f}  rel_err={rel_err:.2e}  "
          f"{'✓' if ok else '✗'}")

print(f"\n  Gradient check: {'PASSED ✓' if all_ok else 'FAILED ✗'}")
if not all_ok:
    print("  ⚠ Check your backward() implementation before continuing.")
print_header("STEP 7 — Training Loop")

def train(X, y, n_epochs=5000, lr=0.5, print_every=500):
    """
    Full training loop with gradient descent.
    Returns trained params and loss history.
    """
    p    = init_params()
    hist = []

    for epoch in range(1, n_epochs + 1):
        # Forward
        a2, cache = forward(X, p)
        loss      = bce_loss(y, a2)
        hist.append(loss)

        # Backward
        grads = backward(y, p, cache)

        # Gradient descent update
        for key in p:
            p[key] -= lr * grads[key]

        if epoch % print_every == 0 or epoch == 1:
            preds    = (a2 > 0.5).astype(int)
            accuracy = (preds == y).mean() * 100
            print(f"  Epoch {epoch:>5}  loss={loss:.6f}  acc={accuracy:.1f}%")

    return p, hist


print(f"  Training on {len(X_train)} noisy XOR samples...")
trained_params, loss_history = train(X_train, y_train,
                                      n_epochs=5000, lr=0.5, print_every=1000)

print_header("STEP 8 — Evaluate on XOR Truth Table")

a2_final, _ = forward(X_xor, trained_params)
preds_final = (a2_final > 0.5).astype(int)

print(f"\n  {'Input':>10}  {'True':>6}  {'Pred_prob':>10}  {'Pred':>6}  {'Correct':>8}")
print("  " + "-" * 48)
for i in range(4):
    x_str   = str(X_xor[i].astype(int).tolist())
    correct = "✓" if preds_final[i, 0] == y_xor[i, 0] else "✗"
    print(f"  {x_str:>10}  {int(y_xor[i,0]):>6}  "
          f"{a2_final[i,0]:>10.4f}  {preds_final[i,0]:>6}  {correct:>8}")

accuracy = (preds_final == y_xor).mean() * 100
print(f"\n  XOR accuracy: {accuracy:.1f}%")
if accuracy == 100.0:
    print("""
  ╔══════════════════════════════════════════╗
  ║   PHASE 1 MILESTONE COMPLETE ✓           ║
  ╚══════════════════════════════════════════╝""")
else:
    print("  ⚠ Network didn't fully converge — try increasing n_epochs or lr.")


print_header("STEP 9 — Loss Curve")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
axes[0].plot(loss_history, color='steelblue', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('BCE Loss')
axes[0].set_title('Training Loss — XOR (manual backprop)')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

# Decision boundary
h  = 0.02
xx = np.arange(-0.3, 1.3, h)
yy = np.arange(-0.3, 1.3, h)
XX, YY = np.meshgrid(xx, yy)
grid   = np.c_[XX.ravel(), YY.ravel()]
ZZ, _  = forward(grid, trained_params)
ZZ     = ZZ.reshape(XX.shape)

axes[1].contourf(XX, YY, ZZ, levels=50, cmap='RdBu', alpha=0.7)
axes[1].contour(XX, YY, ZZ, levels=[0.5], colors='black', linewidths=2)
scatter_colors = ['red' if label == 1 else 'blue' for label in y_xor.ravel()]
axes[1].scatter(X_xor[:, 0], X_xor[:, 1],
                c=scatter_colors, s=200, zorder=5, edgecolors='black')
axes[1].set_title('Decision Boundary — XOR')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), 'xor_training.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n  Plot saved to: {out_path}")
plt.close()


