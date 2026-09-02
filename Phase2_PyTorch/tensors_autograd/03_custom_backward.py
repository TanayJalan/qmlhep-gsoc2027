import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)


class CustomReLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        """
        Apply ReLU and save the input for backward.
        We save x (not the output) because the gradient depends on x.
        """
        ctx.save_for_backward(x)
        return x.clamp(min=0)          # same as torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: dL/d(relu_output), shape same as output
        Returns:     dL/dx = grad_output * (x > 0)
        """
        x, = ctx.saved_tensors
        grad_x = grad_output * (x > 0).float()   # binary mask
        return grad_x


# Test against PyTorch built-in
x = torch.randn(5, requires_grad=True)
print(f"\n  Input x: {x.detach().tolist()}")

# Custom
y_custom  = CustomReLU.apply(x)
y_custom.sum().backward()
grad_custom = x.grad.clone()
x.grad = None                              # zero grad

# Built-in
y_builtin = torch.relu(x)
y_builtin.sum().backward()
grad_builtin = x.grad.clone()

print(f"\n  Custom ReLU output:   {y_custom.detach().tolist()}")
print(f"  Built-in relu output: {y_builtin.detach().tolist()}")
print(f"\n  Custom gradient:      {grad_custom.tolist()}")
print(f"  Built-in gradient:    {grad_builtin.tolist()}")
print(f"\n  Outputs match:   {'✓' if torch.allclose(y_custom, y_builtin) else '✗'}")
print(f"  Gradients match: {'✓' if torch.allclose(grad_custom, grad_builtin) else '✗'}")

# Verify with gradcheck (PyTorch's numerical gradient checker)
x_check  = torch.randn(4, dtype=torch.float64, requires_grad=True)
gradcheck = torch.autograd.gradcheck(CustomReLU.apply, (x_check,), eps=1e-6)
print(f"  gradcheck passed: {'✓' if gradcheck else '✗'}")

#section 3-

class CustomSigmoid(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        sig = torch.sigmoid(x)          # compute sigmoid
        ctx.save_for_backward(sig)      # save OUTPUT, not input
        return sig

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: dL/d(sigma), shape same as output
        Returns:     dL/dx = dL/d(sigma) * sigma*(1-sigma)
        """
        sig, = ctx.saved_tensors
        grad_x = grad_output * sig * (1 - sig)
        return grad_x


# Test
x = torch.randn(5, requires_grad=True)

y_custom = CustomSigmoid.apply(x)
y_custom.sum().backward()
grad_custom = x.grad.clone()
x.grad = None

y_builtin = torch.sigmoid(x)
y_builtin.sum().backward()
grad_builtin = x.grad.clone()

print(f"\n  Custom sigmoid output:   {y_custom.detach().round(decimals=4).tolist()}")
print(f"  Built-in sigmoid output: {y_builtin.detach().round(decimals=4).tolist()}")
print(f"\n  Outputs match:   {'✓' if torch.allclose(y_custom, y_builtin) else '✗'}")
print(f"  Gradients match: {'✓' if torch.allclose(grad_custom, grad_builtin) else '✗'}")

x_check = torch.randn(4, dtype=torch.float64, requires_grad=True)
gc = torch.autograd.gradcheck(CustomSigmoid.apply, (x_check,))
print(f"  gradcheck passed: {'✓' if gc else '✗'}")


class HardThreshold(torch.autograd.Function):
    """
    Forward:  f(x) = 1 if x > 0.5 else 0   (step function)
    Backward: df/dx = 1                      (straight-through)
    """
    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output    # pass gradient through unchanged


# Demo: STE allows a binary layer to receive gradient signal
x = torch.tensor([0.2, 0.6, 0.9, 0.1, 0.8], requires_grad=True)
out = HardThreshold.apply(x)
loss = out.sum()
loss.backward()

print(f"\n  Input x:           {x.detach().tolist()}")
print(f"  Hard threshold:    {out.detach().tolist()}  (0 or 1)")
print(f"  Gradient (STE):    {x.grad.tolist()}")
print(f"  (True derivative would be all zeros — STE lets gradient flow)")




class QuantumCircuitAnalogue(torch.autograd.Function):
    """
    Simulates a parameterised quantum gate expectation value.
    Stand-in: f(θ) = sin(θ)  (like a Rx gate expectation ⟨Z⟩)

    The real circuit would call a quantum simulator here.
    The gradient uses the parameter shift rule — no backprop through the circuit.
    """

    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        # In PennyLane: return qml.expval(qml.PauliZ(0))
        return torch.sin(theta)

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        shift  = torch.tensor(np.pi / 2)

        # Parameter shift rule: [f(θ+π/2) - f(θ-π/2)] / 2
        grad_theta = (torch.sin(theta + shift) -
                      torch.sin(theta - shift)) / 2

        return grad_output * grad_theta   # chain rule with upstream gradient


# Test the parameter shift rule gradient
theta = torch.tensor([0.0, 0.5, 1.0, 1.5], requires_grad=True)

# Our custom (shift rule) gradient
out = QuantumCircuitAnalogue.apply(theta)
out.sum().backward()
grad_shift = theta.grad.clone()

# True gradient of sin(θ) is cos(θ)
grad_true = torch.cos(theta.detach())

print(f"\n  θ values:           {theta.detach().tolist()}")
print(f"\n  f(θ) = sin(θ):      {out.detach().round(decimals=4).tolist()}")
print(f"\n  Parameter shift ∇:  {grad_shift.round(decimals=6).tolist()}")
print(f"  True cos(θ)      ∇:  {grad_true.round(decimals=6).tolist()}")
print(f"\n  Match: {'✓' if torch.allclose(grad_shift, grad_true) else '✗'}")

# gradcheck with float64 for numerical precision
theta_check = torch.tensor([0.3, 0.7, 1.2], dtype=torch.float64, requires_grad=True)
gc = torch.autograd.gradcheck(QuantumCircuitAnalogue.apply, (theta_check,))
print(f"  gradcheck passed: {'✓' if gc else '✗'}")

print("""
  When you write your QVIT in Phase 7:
      1. PennyLane runs the quantum circuit (forward pass)
      2. Parameter shift rule computes the gradient (backward pass)
      3. Both steps are wrapped in a torch.autograd.Function
      4. PyTorch optimizers (Adam, SGD) update the circuit parameters
         exactly like they update classical weights

  The only difference from a classical nn.Linear is what happens
  inside forward() — instead of x @ W.T, it's a quantum circuit.
  The gradient mechanism is the same.
""")



print("SECTION 6 — Custom Function inside nn.Module")

print("""
  Custom autograd.Functions integrate seamlessly with nn.Module.
  You call .apply() inside forward() and PyTorch handles the rest.
""")


class BinaryLayer(nn.Module):
    """
    Linear layer whose activations are binarised (0 or 1) via STE.
    Used in binary neural networks to reduce memory/compute.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        z   = self.linear(x)                    # standard linear
        out = HardThreshold.apply(torch.sigmoid(z))  # binarise output
        return out


model = BinaryLayer(4, 3)
x     = torch.randn(2, 4, requires_grad=True)
out   = model(x)
loss  = out.sum()
loss.backward()

print(f"\n  BinaryLayer output (0s and 1s):\n{out.detach()}")
print(f"  Input gradient (should flow via STE):\n{x.grad.round(decimals=4)}")
print(f"  Gradients are non-zero: "
      f"{'✓' if (x.grad.abs() > 0).any() else '✗'}  "
      f"(STE enabled gradient flow through hard threshold)")

