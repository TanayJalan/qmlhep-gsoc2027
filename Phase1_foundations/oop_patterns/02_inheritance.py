import numpy as np
import sys
import os

np.random.seed(42)

def print_header(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)

class Layer:
    def __init__(self):
        self.training = True
    def forward(self, x):
        raise NotImplementedError
    def __call__(self, x):
        return self.forward(x)
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class LinearLayer(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * scale
        self.b = np.zeros(out_features)
        self._last_input = None
    def forward(self, x):
        self._last_input = x
        return x @ self.W.T + self.b
    def __repr__(self):
        return f"LinearLayer(in={self.in_features}, out={self.out_features})"

class ReLU(Layer):
    def forward(self, x):
        self._last_input = x
        return np.maximum(0, x)

class Sigmoid(Layer):
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class Softmax(Layer):
    def forward(self, x):
        shifted = x - x.max(axis=-1, keepdims=True)
        exps    = np.exp(shifted)
        return exps / exps.sum(axis=-1, keepdims=True)

print_header("Sequential Container")

print("""
  Goal: instead of manually looping over layers like this —
      h = x
      for layer in layers:
          h = layer(h)
  — wrap it in a container so you can write:
      model = Sequential(layer1, layer2, layer3)
      output = model(x)
  and everything else is handled internally.
""")


class Sequential(Layer):
    """
    Chains layers in order: output of layer[i] feeds into layer[i+1].

    Usage:
        model = Sequential(
            LinearLayer(784, 256),
            ReLU(),
            LinearLayer(256, 10),
            Softmax()
        )
        output = model(x)
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pass x through every layer in order."""
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def train(self):
        """Set all layers to training mode."""
        self.training = True
        for layer in self.layers:
            layer.training = True
        return self

    def eval(self):
        """Set all layers to eval mode (disables dropout etc.)."""
        self.training = False
        for layer in self.layers:
            layer.training = False
        return self

    def parameters(self) -> list:
        """Return a list of all (W, b) tuples from LinearLayers."""
        params = []
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                params.append(('W', layer.W, layer))
                params.append(('b', layer.b, layer))
        return params

    def count_params(self) -> int:
        """Total number of scalar parameters."""
        total = 0
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                total += layer.W.size + layer.b.size
        return total

    def summary(self) -> None:
        """Print a clean model summary — like Keras model.summary()."""
        print(f"\n  {'Layer':<30} {'Output shape':<20} {'Params':>8}")
        print("  " + "-" * 60)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, LinearLayer):
                out_shape = f"(batch, {layer.out_features})"
                params    = layer.W.size + layer.b.size
            else:
                out_shape = "same as input"
                params    = 0
            print(f"  {repr(layer):<30} {out_shape:<20} {params:>8,}")
        print("  " + "-" * 60)
        print(f"  {'Total parameters':<50} {self.count_params():>8,}")

    def __repr__(self):
        layer_str = "\n    ".join(repr(l) for l in self.layers)
        return f"Sequential(\n    {layer_str}\n)"


# ── Build and test a model 
print("\n  Building model: 784 → 256 → 128 → 10 (MNIST classifier)")

model = Sequential(
    LinearLayer(784, 256),
    ReLU(),
    LinearLayer(256, 128),
    ReLU(),
    LinearLayer(128, 10),
    Softmax()
)

model.summary()

# Forward pass
batch_size = 32
x = np.random.randn(batch_size, 784)    # 32 flattened MNIST images
output = model(x)

print(f"\n  Input  shape: {x.shape}")
print(f"  Output shape: {output.shape}")
print(f"  First sample output (class probs): {output[0].round(4)}")
print(f"  First sample sums to: {output[0].sum():.8f}")
print(f"  Predicted class: {output[0].argmax()}")


print_header("Parameter Access and Inspection")

print("""
  In PyTorch you call model.parameters() to get all trainable params.
  This is what the optimiser iterates over when updating weights.
  Our version returns (name, array, layer_reference) tuples.
""")

all_params = model.parameters()
print(f"\n  Number of parameter tensors: {len(all_params)}")
for name, param, layer in all_params:
    print(f"    {layer.__class__.__name__} .{name}  "
          f"shape={param.shape}  numel={param.size:,}")

print(f"\n  Total scalar parameters: {model.count_params():,}")


print_header("Train / Eval Mode Toggling")

print("""
  Certain layers (Dropout, BatchNorm) behave differently during
  training vs evaluation. The .train() / .eval() flag controls this.
  Always call model.eval() before running inference.
""")

print(f"\n  Before: model.training = {model.training}")
print(f"  Layers: {[l.training for l in model.layers]}")

model.eval()
print(f"\n  After eval(): model.training = {model.training}")
print(f"  Layers: {[l.training for l in model.layers]}")

model.train()
print(f"\n  After train(): model.training = {model.training}")


print_header("Nested Sequential — Residual-style Block")

print("""
  In a Transformer, you have repeated blocks of:
      attention + feedforward
  You could represent each block as its own Sequential,
  then chain blocks together.

  Here we build a tiny feedforward block as nested Sequential.
""")

def FFN(d_model: int, d_ff: int) -> Sequential:
    """Feedforward sublayer used inside a Transformer block."""
    return Sequential(
        LinearLayer(d_model, d_ff),
        ReLU(),
        LinearLayer(d_ff, d_model)
    )

d_model, d_ff = 64, 256
ffn = FFN(d_model, d_ff)
ffn.summary()

x_tok = np.random.randn(8, d_model)    # 8 tokens, 64-dim embeddings
out   = ffn(x_tok)
print(f"\n  FFN: {x_tok.shape} → {out.shape}  (shape preserved ✓)")
print(f"  Params: {ffn.count_params():,}")
