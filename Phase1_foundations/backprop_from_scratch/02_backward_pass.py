"""
The backward pass — deriving and implementing every gradient by hand.

Architecture reminder:
    z1 = X  @ W1.T + b1
    a1 = tanh(z1)
    z2 = a1 @ W2.T + b2
    a2 = sigmoid(z2)
    L  = BCE(y, a2)

Chain rule — work right to left:
    dL/da2 → dL/dz2 → dL/dW2, dL/db2, dL/da1 → dL/dz1 → dL/dW1, dL/db1

"""

import numpy as np
np.random.seed(42)

# Sigmoid function
def sigmoid(z):
    return 1.0/ (1.0 + np.exp(-np.clip(z, -500, 500)))

def tanh_act(z):
    return np.tanh(z)

def linear_forward(X,W,b):
    return X @ W.T + b

def forward (X, W1, b1, W2, b2):
    z1 = linear_forward(X, W1, b1)
    a1 = tanh_act(z1)
    z2 = linear_forward(a1, W2, b2)
    a2 = sigmoid(z2)
    cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache

def bce_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) +
                          (1 - y_true) * np.log(1 - y_pred)))

def init_weights(n_in, n_out, seed=0):
    np.random.seed(seed)
    return np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)

n_in, n_hidden, n_out = 2, 4, 1
W1 = init_weights(n_in, n_hidden, seed=1)
b1 = np.zeros((1, n_hidden))
W2 = init_weights(n_hidden, n_out, seed=2)
b2 = np.zeros((1, n_out))

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)

# Section 1- Chain rule

# Section 2- Derivative of BCE+Sigmoid

print("SECTION 2 — BCE + Sigmoid Gradient")

print("""
  The gradient dL/dz2 = a2 - y is a famous result.
  Let's derive it step by step so it's not magic.

  BCE loss (for one sample):
      L = -[y * log(a2) + (1-y) * log(1-a2)]

  dL/da2:
      = -[y/a2 - (1-y)/(1-a2)]

  sigmoid derivative:
      da2/dz2 = a2 * (1 - a2)

  Chain rule:
      dL/dz2 = dL/da2 * da2/dz2
             = -[y/a2 - (1-y)/(1-a2)] * a2 * (1-a2)
             = -y*(1-a2) + (1-y)*a2
             = a2 - y    √

  This is why sigmoid + BCE is the standard binary classifier:
  the gradient is simply prediction minus truth. Clean and stable.
""")


#verify numerically
a2_test, cache_test = forward(X_xor, W1, b1, W2, b2)
N = len(X_xor)

# Our formula: dz2 = (a2 - y) / N
dz2_formula = (a2_test - y_xor) / N

# Numerical: perturb z2 and finite difference
eps     = 1e-5
dz2_num = np.zeros_like(cache_test['z2'])
for i in range(N):
    z2_plus  = cache_test['z2'].copy(); z2_plus[i]  += eps
    z2_minus = cache_test['z2'].copy(); z2_minus[i] -= eps
    L_plus   = bce_loss(y_xor, sigmoid(z2_plus))
    L_minus  = bce_loss(y_xor, sigmoid(z2_minus))
    dz2_num[i] = (L_plus - L_minus) / (2 * eps)

print(f"  dz2 (formula):   {dz2_formula.T.round(6)}")
print(f"  dz2 (numerical): {dz2_num.T.round(6)}")
match = np.allclose(dz2_formula, dz2_num, atol=1e-5)
print(f"\n  Match: {'√' if match else 'wrong'}")

# Section 3- Output layer gradients( dw2, db2)

print("SECTION 3 — Output Layer Gradients (dW2, db2)")

print("""
  z2 = a1 @ W2.T + b2
  We have dL/dz2. Now:

  dL/dW2:
      z2[i,j] = Σ_k a1[i,k] * W2[j,k] + b2[0,j]
      dL/dW2[j,k] = Σ_i dz2[i,j] * a1[i,k]
                  = dz2.T @ a1       shape: (n_out, n_hid)

  dL/db2:
      z2[i,j] = ... + b2[0,j]
      dL/db2[0,j] = Σ_i dz2[i,j]   (sum over batch)
                  = dz2.sum(axis=0)  shape: (1, n_out)
""")

dz2 = (a2_test - y_xor) / N   # (4, 1)

dW2 = dz2.T @ cache_test['a1']                    # (1, 4)
db2 = dz2.sum(axis=0, keepdims=True)               # (1, 1)

print(f"  dz2 shape: {dz2.shape}")
print(f"  a1  shape: {cache_test['a1'].shape}")
print(f"  dW2 = dz2.T @ a1  →  shape: {dW2.shape}")
print(f"  db2 = dz2.sum(axis=0)  →  shape: {db2.shape}")
print(f"\n  dW2: {dW2.round(6)}")
print(f"  db2: {db2.round(6)}")


# Section 4- Backprop Through hidden layer (dw1, db1)

print("""
  dL/da1:
      z2 = a1 @ W2.T + b2
      dL/da1[i,k] = Σ_j dz2[i,j] * W2[j,k]
                  = dz2 @ W2        shape: (batch, n_hid)

  Tanh derivative:
      d/dz tanh(z) = 1 - tanh(z)^2 = 1 - a1^2
      (a1 = tanh(z1), so we already have it in cache)

  dL/dz1:
      = dL/da1 * (1 - a1^2)         element-wise, shape: (batch, n_hid)

  dL/dW1:
      z1 = X @ W1.T + b1
      dL/dW1 = dz1.T @ X            shape: (n_hid, n_in)

  dL/db1:
      = dz1.sum(axis=0)             shape: (1, n_hid)
""")

da1 = dz2 @ W2                                # (4, 4) — error flows back through W2
dz1 = da1 * (1 - cache_test['a1'] ** 2)       # (4, 4) — gated by tanh'

dW1 = dz1.T @ cache_test['X']                 # (4, 2)
db1 = dz1.sum(axis=0, keepdims=True)           # (1, 4)

print(f"  dz2 shape: {dz2.shape}  W2 shape: {W2.shape}")
print(f"  da1 = dz2 @ W2              →  shape: {da1.shape}")
print(f"  dz1 = da1 * (1 - a1^2)     →  shape: {dz1.shape}")
print(f"  dW1 = dz1.T @ X            →  shape: {dW1.shape}")
print(f"  db1 = dz1.sum(axis=0)      →  shape: {db1.shape}")
print(f"\n  dW1:\n{dW1.round(6)}")
print(f"  db1: {db1.round(6)}")


# Section 5 - Full backward() function

print("Section 5")

def backward(y_true: np.ndarray,
             W2: np.ndarray,
             cache: dict) -> dict:

             a2 = cache['a2']
             a1 = cache['a1']
             X  = cache['X']
             N  = len(y_true)

             # Output layer
             dz2 = (a2 - y_true) / N           # combined sigmoid + BCE gradient
             dW2 = dz2.T @ a1
             db2 = dz2.sum(axis=0, keepdims=True)

             # Hidden layer
             da1 = dz2 @ W2                    # propagate error back through W2
             dz1 = da1 * (1 - a1 ** 2)        # gate by tanh derivative
             dW1 = dz1.T @ X
             db1 = dz1.sum(axis=0, keepdims=True)
 
             return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}


grads = backward(y_xor, W2, cache_test)
print(f"\n  backward() returned:")
for key, val in grads.items():
    print(f"    d{key}: shape={val.shape}  "
          f"mean={val.mean():.6f}  max_abs={np.abs(val).max():.6f}")


# Section 6- NUmerical Gradientcheck ( all parameter)
print("SECTION 6 — Full Numerical Gradient Check")

print("""
  Finite difference approximation:
      dL/dθ ≈ [L(θ+ε) - L(θ-ε)] / (2ε)

  We check every single element of W1, b1, W2, b2.
  If relative error < 1e-4, backprop is correct.

  This is a non-negotiable sanity check.
  In serious research code, this check lives in the test suite.
""")


eps = 1e-5

def numerical_grad_full(X, y, W1, b1, W2, b2):
    """Compute numerical gradient for every parameter."""
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    num_grads = {}

    for key, P in params.items():
        grad = np.zeros_like(P)
        it   = np.nditer(P, flags=['multi_index'])

        while not it.finished:
            idx = it.multi_index

            orig = P[idx]
            P[idx] = orig + eps
            a2_plus, _ = forward(X, W1, b1, W2, b2)
            L_plus = bce_loss(y, a2_plus)

            P[idx] = orig - eps
            a2_minus, _ = forward(X, W1, b1, W2, b2)
            L_minus = bce_loss(y, a2_minus)

            P[idx] = orig   # restore
            grad[idx] = (L_plus - L_minus) / (2 * eps)
            it.iternext()

        num_grads[key] = grad

    return num_grads

# Run analytic backward
a2_chk, cache_chk = forward(X_xor, W1, b1, W2, b2)
grads_analytic     = backward(y_xor, W2, cache_chk)

# Run numerical gradient
grads_numeric = numerical_grad_full(X_xor, y_xor, W1, b1, W2, b2)

print(f"\n  {'Param':<6}  {'Max rel error':>14}  {'Status':>8}")
print(f"  {'-'*34}")

all_ok = True
for key in ['W1', 'b1', 'W2', 'b2']:
    an  = grads_analytic[key]
    num = grads_numeric[key]
    rel_err = (np.abs(an - num) /
               (np.abs(an) + np.abs(num) + 1e-15)).max()
    ok = rel_err < 1e-4
    all_ok = all_ok and ok
    print(f"  {key:<6}  {rel_err:>14.2e}  {'PASS' if ok else 'FAIL':>8}")

print(f"\n  Overall: {'ALL PASSED' if all_ok else 'FAILURES — check backward()'}")
if not all_ok:
    print("\n  Debug tips:")
    print("  1. Check dz2: should be (a2-y)/N")
    print("  2. Check da1: should be dz2 @ W2 (not W2.T)")
    print("  3. Check dz1: should multiply by (1-a1^2), not (1-a1)")
    print("  4. Check dW shapes match W shapes exactly")


# Section 7- Gradient Intuition

print("SECTION 7 — What Gradients Tell You")

print("""
  A gradient dL/dW[i,j] answers:
      "If I increase W[i,j] by a tiny amount,
       how much does the loss increase?"

  Large positive gradient → decreasing W[i,j] reduces loss
  Large negative gradient → increasing W[i,j] reduces loss
  Gradient ≈ 0           → W[i,j] barely affects the loss
                           (either it's already optimal, or it's dead)

  Gradient descent update:
      W = W - lr * dL/dW

  This is exactly what 03_gradient_descent.py implements.
""")

print(f"\n  Current gradients (XOR, random weights):")
for key, g in grads_analytic.items():
    max_i = np.unravel_index(np.abs(g).argmax(), g.shape)
    print(f"    d{key}  max abs grad = {np.abs(g).max():.6f}  "
          f"at index {max_i}  "
          f"(positive → decrease {key}{list(max_i)} to reduce loss)")

