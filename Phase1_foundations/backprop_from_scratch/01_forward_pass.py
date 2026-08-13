"""
The forward pass only — no training, no gradients yet.
We build the 2-layer network architecture as pure NumPy
and verify each layer's output shape and value range.

Architecture:
    Input  (N, 2)
      ↓  Linear(2 → 4):   z1 = X @ W1.T + b1
      ↓  Tanh:             a1 = tanh(z1)
      ↓  Linear(4 → 1):   z2 = a1 @ W2.T + b2
      ↓  Sigmoid:          a2 = sigma(z2)
    Output (N, 1)  — probability of class 1

This is the same network we will train in 04_two_layer_net.py.
Here we only care that shapes are right and activations are bounded.

Forward links:
    02_backward_pass.py  — we add gradients to this forward pass
    04_two_layer_net.py  — we add the training loop

"""


from numpy import ndarray
import numpy as np

np.random.seed(42)

# Section 1 -Activation Functions

print("""
  We implement each activation from its mathematical definition.
  In 02_backward_pass.py we will need their derivatives —
  so understanding the function first is essential.
""")

# applying sigmoid function | sigma(z) = 1 / (1 + e^{-z})
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0/ (1.0 + np.exp(-np.clip(z,-500,500)))

# tanh(z) = (e^z - e^{-z}) / (e^z + e^{-z})
def tanh_act(z : np.ndarray) -> np.ndarray:
    return np.tanh(z)

# softmax(z)_i = e^{z_i} / Σ_j e^{z_j}
def softmax(z: np.ndarray) -> np.ndarray:
    shifted = z- z.max(axis =-1, keepdims = True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis =-1, keepdims= True)

# ReLU(z) = max(0, z)
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0,z)

# Test all activities
test_z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
print(f"\n  Test input z: {test_z}")
print(f"  sigmoid(z):   {sigmoid(test_z).round(4)}")
print(f"  tanh(z):      {tanh_act(test_z).round(4)}")
print(f"  relu(z):      {relu(test_z).round(4)}")

assert (sigmoid(test_z) > 0).all() and (sigmoid(test_z) < 1).all()
print(f"\n sigmoid output in (0,1)")
assert (tanh_act(test_z) > -1).all() and (tanh_act(test_z) < 1).all()
print(f" tanh output in (-1, 1)")
z_batch = np.random.randn(4, 5)
assert np.allclose(softmax(z_batch).sum(axis=1), 1.0)
print(f" softmax rows sum to 1.0")

# Section 2 - Weight Initialisation

print("""
  All-zero weights:  every neuron computes the same thing.
                     symmetry is never broken — network stays useless.
  Too large weights: activations saturate -> gradients vanish.
  Too small weights: signal shrinks to zero through layers.

  Xavier / Glorot init (for tanh networks):
      W ~ Normal(0, √(1 / n_in))
  Keeps activation variance stable through depth.

  Kaiming / He init (for ReLU networks):
      W ~ Normal(0, sqrt(2 / n_in))
  Factor of 2 compensates for ReLU zeroing half its inputs.
""")


def init_weights(n_in: int, n_out: int, seed: int = 0) -> np.ndarray:
    """normal init. Shape: (n_out, n_in)."""
    np.random.seed(seed)
    scale = np.sqrt(1.0 / n_in)
    return np.random.randn(n_out, n_in) * scale


def init_bias(n_out: int) -> np.ndarray:
    """Bias initialised to zero. Shape: (1, n_out)."""
    return np.zeros((1, n_out))


n_in, n_hidden, n_out = 2, 4, 1

W1 = init_weights(n_in,     n_hidden, seed=1)   # (4, 2)
b1 = init_bias(n_hidden)                          # (1, 4)
W2 = init_weights(n_hidden, n_out,    seed=2)   # (1, 4)
b2 = init_bias(n_out)                             # (1, 1)

print(f"\n  W1  shape={W1.shape}  mean={W1.mean():.4f}  std={W1.std():.4f}")
print(f"  b1  shape={b1.shape}  values={b1}")
print(f"  W2  shape={W2.shape}  mean={W2.mean():.4f}  std={W2.std():.4f}")
print(f"  b2  shape={b2.shape}  values={b2}")
print(f"\n  Expected W1 std ≈ {np.sqrt(1/n_in):.4f}  (actual: {W1.std():.4f})")
print(f"  Expected W2 std ≈ {np.sqrt(1/n_hidden):.4f}  (actual: {W2.std():.4f})")


# Section 3 - Linear Layer (forward only)

def linear_forward(X: np.ndarray,
                    W: np.ndarray,
                    b: np.ndarray
                    ) -> np.ndarray:
                    return X @ W.T + b

X_xor = np.array([[0,0],
                [0,1],
                [1,0],
                [1,1]], dtype=float)

y_xor = np.array([[0], [1], [1], [0]], dtype=float)

z1_test = linear_forward(X_xor, W1, b1)
print(f"\n  X_xor shape: {X_xor.shape}")
print(f"  z1 = X @ W1.T + b1  ->  shape: {z1_test.shape}")
print(f"  z1:\n{z1_test.round(4)}")

# Section 4 - Full forward pass

def forward(X: np.ndarray,
            W1: np.ndarray, b1: np.ndarray,
            W2: np.ndarray, b2: np.ndarray) -> tuple:
            z1 = linear_forward(X, W1, b1)    # (batch, 4)
            a1 = tanh_act(z1)                  # (batch, 4)
            z2 = linear_forward(a1, W2, b2)   # (batch, 1)
            a2 = sigmoid(z2)                   # (batch, 1)

            cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
            return a2, cache


a2, cache = forward(X_xor, W1, b1, W2, b2)

print(f"\n  Forward pass on XOR (random weights — output is meaningless):")
print(f"\n  {'Input':>10}  {'True':>6}  {'Pred prob':>10}  {'Pred class':>11}")
print(f"  {'-'*44}")
for i in range(4):
    pred_class = int(a2[i, 0] > 0.5)
    print(f"  {str(X_xor[i].astype(int).tolist()):>10}  "
          f"{int(y_xor[i,0]):>6}  "
          f"{a2[i,0]:>10.4f}  "
          f"{pred_class:>11}")

# section 5- shape checks

print("SECTION 5 — Shape Checks (all batch sizes)")

for bs in [1, 4, 16, 64]:
    X_t  = np.random.randn(bs, n_in)
    a2_t, c = forward(X_t, W1, b1, W2, b2)

    assert c['z1'].shape == (bs, n_hidden)
    assert c['a1'].shape == (bs, n_hidden)
    assert c['z2'].shape == (bs, n_out)
    assert a2_t.shape    == (bs, n_out)
    assert (a2_t > 0).all() and (a2_t < 1).all()

    print(f"  batch={bs:>3}  {X_t.shape} "
          f"→ z1:{c['z1'].shape} "
          f"→ a1:{c['a1'].shape} "
          f"→ a2:{a2_t.shape}")

print(f"\n  All shape checks passed ")

# Section 6 - Loss function

print("""
  BCE(y, ŷ) = -1/N * Σ [ y*log(ŷ) + (1-y)*log(1-ŷ) ]

  Random binary classifier baseline: log(2) ≈ 0.693
  Perfect predictions: loss → 0
  Completely wrong:    loss → ∞
""")

def bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary cross-entropy. Clips y_pred to avoid log(0)."""
    eps    = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    ))


loss_random = bce_loss(y_xor, a2)
loss_perfect = bce_loss(y_xor, np.array([[0.001],[0.999],[0.999],[0.001]]))
loss_wrong   = bce_loss(y_xor, np.array([[0.999],[0.001],[0.001],[0.999]]))

print(f"\n  BCE (random weights):  {loss_random:.4f}")
print(f"  BCE (near-perfect):    {loss_perfect:.4f}")
print(f"  BCE (totally wrong):   {loss_wrong:.4f}")
print(f"  Random baseline log2:  {np.log(2):.4f}")

