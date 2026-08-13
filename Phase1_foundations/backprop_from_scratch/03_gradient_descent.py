"""
Covers:
    1. Vanilla gradient descent (single batch)
    2. Learning rate sensitivity — too high, too low, just right
    3. Mini-batch gradient descent
    4. Momentum — the simplest improvement to vanilla GD
    5. Watching the loss decrease in real time

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(42)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def tanh_act(z):
    return np.tanh(z)

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1.T + b1
    a1 = tanh_act(z1)
    z2 = a1 @ W2.T + b2
    a2 = sigmoid(z2)
    return a2, {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

def bce_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) +
                          (1 - y_true) * np.log(1 - y_pred)))

def backward(y_true, W2, cache):
    a2, a1, X = cache['a2'], cache['a1'], cache['X']
    N   = len(y_true)
    dz2 = (a2 - y_true) / N
    dW2 = dz2.T @ a1
    db2 = dz2.sum(axis=0, keepdims=True)
    da1 = dz2 @ W2
    dz1 = da1 * (1 - a1 ** 2)
    dW1 = dz1.T @ X
    db1 = dz1.sum(axis=0, keepdims=True)
    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

def init_params(seed=0):
    np.random.seed(seed)
    return {
        'W1': np.random.randn(4, 2) * np.sqrt(1.0 / 2),
        'b1': np.zeros((1, 4)),
        'W2': np.random.randn(1, 4) * np.sqrt(1.0 / 4),
        'b2': np.zeros((1, 1)),
    }

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)


# section 1 - vanilla gradient descent 

print("SECTION 1 — One Step of Gradient Descent")

print("""
  Gradient descent update rule:
      θ ← θ - lr * dL/dθ

  For each parameter:
      W1 ← W1 - lr * dW1
      b1 ← b1 - lr * db1
      W2 ← W2 - lr * dW2
      b2 ← b2 - lr * db2

  lr (learning rate) controls step size.
  Too large → overshoot, loss bounces or diverges.
  Too small → takes forever to converge.
""")

params = init_params(seed=0)

# Loss before
a2, cache = forward(X_xor, **params)
loss_before = bce_loss(y_xor, a2)

# Compute gradients
grads = backward(y_xor, params['W2'], cache)

# One gradient descent step
lr = 0.5
for key in params:
    params[key] = params[key] - lr * grads[key]

# Loss after
a2, _ = forward(X_xor, **params)
loss_after = bce_loss(y_xor, a2)

print(f"\n Learning rate: {lr}")
print(f"Loss before: {loss_before:.6f}")
print(f"Loss after: {loss_after:.6f}")
print(f"Reduction: {loss_before - loss_after:.6f} "
      f"({'loss went down' if loss_after < loss_before else 'loss went up'})")

# Section 2- learning rate sensitivity 

print("SECTION 2 — Learning Rate Sensitivity")

def run_gd(lr, n_steps=500, seed=0):
    p    = init_params(seed=seed)
    hist = []
    for _ in range(n_steps):
        a2, cache = forward(X_xor, **p)
        loss = bce_loss(y_xor, a2)
        hist.append(loss)
        if not np.isfinite(loss) or loss > 100:
            hist += [float('nan')] * (n_steps - len(hist))
            break
        grads = backward(y_xor, p['W2'], cache)
        for key in p:
            p[key] -= lr * grads[key]
    return hist, p

learning_rates = [0.001, 0.1, 0.5, 5.0]
results = {}

print(f"\n  {'LR':>6}  {'Final loss':>12}  {'Converged':>10}")
print(f"  {'-'*34}")
for lr in learning_rates:
    hist, final_p = run_gd(lr, n_steps=1000)
    final_loss    = hist[-1] if np.isfinite(hist[-1]) else float('inf')
    converged     = np.isfinite(final_loss) and final_loss < 0.1
    results[lr]   = (hist, final_p)
    print(f"  {lr:>6}  {final_loss:>12.6f}  "
          f"{'yes' if converged else 'no':>10}")

# Section 3- Mini-batch Gradient Descent

print("SECTION 3 — Mini-batch Gradient Descent")

np.random.seed(1)
N_train   = 200
X_train   = np.tile(X_xor, (N_train // 4, 1)) + 0.1 * np.random.randn(N_train, 2)
y_train   = np.tile(y_xor, (N_train // 4, 1))
perm      = np.random.permutation(N_train)
X_train   = X_train[perm]
y_train   = y_train[perm]

def run_minibatch_gd(X, y, batch_size, lr, n_epochs=50, seed=0):
    """Mini-batch gradient descent. Returns loss per epoch."""
    p         = init_params(seed=seed)
    n_samples = len(X)
    hist      = []

    for epoch in range(n_epochs):
        # Shuffle at start of each epoch
        idx = np.random.permutation(n_samples)
        X_s, y_s = X[idx], y[idx]

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_samples, batch_size):
            X_b = X_s[start:start + batch_size]
            y_b = y_s[start:start + batch_size]

            a2, cache = forward(X_b, **p)
            loss      = bce_loss(y_b, a2)
            grads     = backward(y_b, p['W2'], cache)

            for key in p:
                p[key] -= lr * grads[key]

            epoch_loss += loss
            n_batches  += 1

        hist.append(epoch_loss / n_batches)

    return hist, p


print(f"\n  Running mini-batch GD on {N_train} noisy XOR samples:")
print(f"\n  {'Batch size':>12}  {'Final loss':>12}  {'Acc':>8}")
print(f"  {'-'*36}")

for bs in [1, 16, 64, N_train]:
    hist, final_p = run_minibatch_gd(X_train, y_train,
                                      batch_size=bs, lr=0.3, n_epochs=100)
    a2_eval, _   = forward(X_xor, **final_p)
    loss_eval    = bce_loss(y_xor, a2_eval)
    acc          = ((a2_eval > 0.5).astype(int) == y_xor).mean() * 100
    label        = "SGD" if bs == 1 else ("Full batch" if bs == N_train else f"Mini-batch")
    print(f"  {f'{label} (bs={bs})':>12}  {loss_eval:>12.6f}  {acc:>7.1f}%")


# Section 4- Momentum

print("SECTION 4 — Momentum")

def run_momentum(X, y, lr, beta=0.9, n_steps=500, seed=0):
    p    = init_params(seed=seed)
    v    = {key: np.zeros_like(val) for key, val in p.items()}
    hist = []

    for _ in range(n_steps):
        a2, cache = forward(X, **p)
        loss      = bce_loss(y, a2)
        hist.append(loss)

        if not np.isfinite(loss):
            break

        grads = backward(y, p['W2'], cache)

        for key in p:
            v[key] = beta * v[key] - lr * grads[key]   # update velocity
            p[key] = p[key] + v[key]                    # update params

    return hist, p


print(f"\n  Comparing GD vs Momentum (500 steps, lr=0.1):\n")

hist_gd,  _ = run_gd(lr=0.1, n_steps=500)
hist_mom, _ = run_momentum(X_xor, y_xor, lr=0.1, beta=0.9, n_steps=500)

checkpoints = [10, 50, 100, 200, 500]
print(f"  {'Step':>6}  {'GD loss':>10}  {'Momentum loss':>14}  {'Speedup':>8}")
print(f"  {'-'*44}")
for step in checkpoints:
    idx     = step - 1
    gd_l    = hist_gd[idx]  if idx < len(hist_gd)  else float('nan')
    mom_l   = hist_mom[idx] if idx < len(hist_mom) else float('nan')
    speedup = f"{gd_l/mom_l:.1f}x" if np.isfinite(gd_l) and np.isfinite(mom_l) and mom_l > 0 else "—"
    print(f"  {step:>6}  {gd_l:>10.6f}  {mom_l:>14.6f}  {speedup:>8}")

# Section 5- Plot Loss Curve 

print("SECTION 5 — Plotting Loss Curves")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Learning rate comparison
for lr in learning_rates:
    hist, _ = results[lr]
    valid   = [v for v in hist if np.isfinite(v)]
    axes[0].plot(valid, label=f"lr={lr}", linewidth=1.5)
axes[0].set_xlabel("Step")
axes[0].set_ylabel("BCE Loss")
axes[0].set_title("Learning Rate Sensitivity")
axes[0].legend()
axes[0].set_ylim(0, 1.5)
axes[0].grid(True, alpha=0.3)

# Plot 2: Mini-batch sizes
colors = ['steelblue', 'orange', 'green', 'red']
for i, bs in enumerate([1, 16, 64, N_train]):
    hist, _ = run_minibatch_gd(X_train, y_train, batch_size=bs,
                                lr=0.3, n_epochs=100)
    label = "SGD (bs=1)" if bs == 1 else \
            f"Full batch (bs={N_train})" if bs == N_train else \
            f"Mini-batch (bs={bs})"
    axes[1].plot(hist, label=label, color=colors[i], linewidth=1.5)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("BCE Loss")
axes[1].set_title("Mini-batch Size Comparison")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# Plot 3: GD vs Momentum
axes[2].plot(hist_gd,  label="Vanilla GD (lr=0.1)",  linewidth=1.5)
axes[2].plot(hist_mom, label="Momentum β=0.9",        linewidth=1.5)
axes[2].set_xlabel("Step")
axes[2].set_ylabel("BCE Loss")
axes[2].set_title("Gradient Descent vs Momentum")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'gradient_descent_curves.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n  Plots saved to: {out_path}")
plt.close()
