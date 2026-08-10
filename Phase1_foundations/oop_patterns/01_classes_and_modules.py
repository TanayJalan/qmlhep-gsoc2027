import numpy as np

np.random.seed(42)


def print_header(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)

print_header("BASE CLASS — Layer")

class Layer:
    def __init__(self):
        self.training = True          # flag used by dropout/batchnorm later

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()"
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

print_header("LinearLayer — y = x @ W.T + b")

class LinearLayer(Layer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Kaiming uniform initialisation (good default for ReLU networks)
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * scale
        self.b = np.zeros(out_features)

        # Store last input for backprop (we'll use this in backprop_from_scratch)
        self._last_input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_input = x          # for backprop
        return x @ self.W.T + self.b
    
    def __repr__(self):
        return f"LinearLayer(in={self.in_features}, out={self.out_features})"
# Test LinearLayer
batch, in_f, out_f = 4, 8, 3
lin = LinearLayer(in_f, out_f)
x   = np.random.randn(batch, in_f)
out = lin(x)                                 # __call__ → forward

print(f"  LinearLayer: {lin}")
print(f"  Input  shape: {x.shape}")
print(f"  Output shape: {out.shape}")
print(f"  W shape:      {lin.W.shape}")
print(f"  b shape:      {lin.b.shape}")
print(f"  Output (first sample): {out[0].round(4)}")

# Verify manually
expected = x @ lin.W.T + lin.b
assert np.allclose(out, expected), "LinearLayer output mismatch"
print("  ✓ Output matches manual x @ W.T + b")


# ═══════════════════════════════════════════════════════════
# ACTIVATION FUNCTIONS
# ═══════════════════════════════════════════════════════════
print_header("Activation Functions — ReLU, Sigmoid, Tanh")

print("""
  Activations introduce non-linearity — without them, stacking
  linear layers is just one big linear transformation.
  Each activation is itself a Layer subclass.
""")

class ReLU(Layer):
    """
    ReLU(x) = max(0, x)

    Most common activation in modern networks.
    Dead neuron problem: if a neuron always gets negative input,
    its gradient is always 0 and it never updates.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_input = x
        return np.maximum(0, x)


class Sigmoid(Layer):
    """
    Sigmoid(x) = 1 / (1 + exp(-x))  →  output in (0, 1)

    Used in binary classifiers and gates (LSTM, GRU).
    Saturates for large |x|, causing vanishing gradients.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_input = x
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))   # clip for stability


class Tanh(Layer):
    """
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))  →  output in (-1, 1)

    Used in RNNs. Zero-centred (unlike Sigmoid), which helps gradient flow.
    Still saturates for large |x|.
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_input = x
        return np.tanh(x)


class Softmax(Layer):
    """
    Softmax(x)_i = exp(x_i) / sum_j exp(x_j)

    Converts logits to probabilities. Always applied row-wise over a batch.
    Numerically stable: subtract row max before exp().
    """
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_input = x
        shifted = x - x.max(axis=-1, keepdims=True)
        exps    = np.exp(shifted)
        return exps / exps.sum(axis=-1, keepdims=True)


# Test activations on the same input
test_input = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

activations = [ReLU(), Sigmoid(), Tanh(), Softmax()]
for act in activations:
    out = act(test_input)
    print(f"\n  {act.__class__.__name__}:")
    print(f"    input:  {test_input[0]}")
    print(f"    output: {out[0].round(4)}")

# Softmax sums to 1
sm_out = Softmax()(test_input)
print(f"\n  Softmax row sum: {sm_out.sum(axis=1)[0]:.8f}  (should be 1.0)")


def count_params(layers: list) -> int:
    """Count total number of scalar parameters across all LinearLayers."""
    total = 0
    for layer in layers:
        if isinstance(layer, LinearLayer):
            total += layer.W.size + layer.b.size
    return total

# Build a small network manually
network_layers = [
    LinearLayer(784, 256),    # input → hidden1
    ReLU(),
    LinearLayer(256, 128),    # hidden1 → hidden2
    ReLU(),
    LinearLayer(128, 10),     # hidden2 → output
    Softmax()
]

total = count_params(network_layers)
print(f"\n  Network: 784 → 256 → 128 → 10")
for layer in network_layers:
    if isinstance(layer, LinearLayer):
        params = layer.W.size + layer.b.size
        print(f"    {layer}  →  {params:,} params")
print(f"\n  Total trainable parameters: {total:,}")

# Full forward pass through the network manually
x_in = np.random.randn(16, 784)     # batch of 16 MNIST images (flattened)
h = x_in
for layer in network_layers:
    h = layer(h)
print(f"\n  Forward pass: {x_in.shape} → {h.shape}")
print(f"  Output (first sample, should sum to 1.0): {h[0].round(4)}")
print(f"  Row sum: {h[0].sum():.8f}")
