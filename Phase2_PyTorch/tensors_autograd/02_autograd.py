"""
PyTorch's autograd is the Phase 1 backward pass — automated.
Instead of deriving dL/dW by hand, PyTorch builds a computation
graph as operations run and calls .backward() to walk it in reverse.

This file covers:
    1. requires_grad and the computation graph
    2. .backward() and .grad
    3. Verifying autograd matches Phase 1 manual gradients
    4. torch.no_grad() — when to turn off gradient tracking
    5. .detach() — cut a tensor out of the graph
    6. Gradient accumulation and zeroing

Phase 1 connection:
    In 04_two_layer_net.py you derived dL/dW2 = dz2.T @ a1 by hand.
    Here, autograd computes that same value automatically.
    We verify they match — same number, different method.

"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(0)

# Section 1- Requires grad

print('Section 1')

w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([3.0])                    # input — no grad needed

print(f"  w: {w}  requires_grad={w.requires_grad}  is_leaf={w.is_leaf}")
print(f"  b: {b}  requires_grad={b.requires_grad}  is_leaf={b.is_leaf}")
print(f"  x: {x}  requires_grad={x.requires_grad}  is_leaf={x.is_leaf}")

y = w * x + b
print(f"\n  y = w*x + b = {y.item()}")
print(f"  y.requires_grad: {y.requires_grad} inherited from w and b")
print(f"  y.grad_fn: {y.grad_fn} records what created y")
print(f"  y.is_leaf: {y.is_leaf} y was created by an op, not by user")

#section 2- .backward() and .grad

print('Section 2')

w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([3.0])

y = w * x + b          # y = 2*3 + 1 = 7
loss = y ** 2          # L = 49

loss.backward()

print(f"  y = w*x + b = {y.item()}")
print(f"  L = y^2 = {loss.item()}")
print(f"\n  dL/dw (autograd): {w.grad.item()}")
print(f"  dL/dw (manual):   {(2 * y * x).item()} ")
print(f"  dL/db (autograd): {b.grad.item()}")
print(f"  dL/db (manual):   {(2 * y).item()}")
assert w.grad.item() == 2 * y.item() * x.item()
print(f"\n autograd matches manual gradient")

# Section -3 
print('Section 3')

torch.manual_seed(42)

# Identical initialisation to Phase 1
n_in, n_hid, n_out = 2, 4, 1

W1 = (torch.randn(n_hid, n_in) * (2.0/n_in)**0.5).requires_grad_()
b1 = torch.zeros(1,     n_hid, requires_grad=True)
W2 = (torch.randn(n_out, n_hid) * (2.0/n_hid)**0.5).requires_grad_()
b2 = torch.zeros(1,     n_out, requires_grad=True)

# XOR data
X_xor = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y_xor = torch.tensor([[0],[1],[1],[0]],           dtype=torch.float32)

# Forward pass — identical math to Phase 1
z1 = X_xor @ W1.T + b1
a1 = torch.tanh(z1)
z2 = a1 @ W2.T + b2
a2 = torch.sigmoid(z2)

# BCE loss (manual, same formula as Phase 1)
eps  = 1e-15
loss = -torch.mean(
    y_xor * torch.log(a2 + eps) + (1 - y_xor) * torch.log(1 - a2 + eps)
)


print(f"  Loss: {loss.item():.6f}")

# Autograd backward
loss.backward()

print(f"\n  Gradients computed by autograd:")
print(f"    dL/dW1 shape: {W1.grad.shape}  values:\n{W1.grad.round(decimals=6)}")
print(f"    dL/dW2 shape: {W2.grad.shape}  values: {W2.grad}")

# Now compute the same gradients manually (Phase 1 method)
with torch.no_grad():                          # don't track these ops
    N    = len(X_xor)
    dz2  = (a2 - y_xor) / N                   
    dW2_manual = dz2.T @ a1                   
    da1  = dz2 @ W2
    dz1  = da1 * (1 - a1 ** 2)
    dW1_manual = dz1.T @ X_xor               

print(f"\n  Gradients computed manually (Phase 1 method):")
print(f" dL/dW1:\n{dW1_manual.round(decimals=6)}")
print(f" dL/dW2: {dW2_manual}")
match_W1 = torch.allclose(W1.grad, dW1_manual, atol=1e-5)
match_W2 = torch.allclose(W2.grad, dW2_manual, atol=1e-5)

print(f"\n  W1 grads match: {'✓' if match_W1 else '✗'}")
print(f"  W2 grads match: {'✓' if match_W2 else '✗'}")

# section -4
print('section 4')

w = torch.tensor([1.0], requires_grad=True)

for step in range(3):
    y    = (w * 2) ** 2      # L = (2w)^2,  dL/dw = 8w = 8
    y.backward()
    print(f"  Step {step+1}: w.grad = {w.grad.item():.1f}  "
          f"({'accumulates!' if step > 0 else 'first pass'})")

print(f"\n  After 3 backward() calls: w.grad = {w.grad.item():.1f}  (should be 8, got 24)")

# Correct pattern: zero before each backward
w.grad = None
for step in range(3):
    if w.grad is not None:
        w.grad.zero_()
    y = (w * 2) ** 2
    y.backward()
    print(f"  Step {step+1} (with zero_grad): w.grad = {w.grad.item():.1f}")


#section 5
print('section 5')
w = torch.tensor([2.0], requires_grad=True)

# Inside no_grad: operations don't get tracked
with torch.no_grad():
    y = w * 3         # no grad_fn recorded
    print(f"  Inside no_grad:   y.requires_grad = {y.requires_grad}")

# Outside no_grad: tracked normally
y2 = w * 3
print(f"  Outside no_grad:  y2.requires_grad = {y2.requires_grad}")
print(f"  y2.grad_fn:       {y2.grad_fn}")

# Decorator form (for functions)
@torch.no_grad()
def predict(model_fn, x):
    return model_fn(x)

#section 6
print('section 6')

w = torch.tensor([3.0], requires_grad=True)
y = w ** 2 + 2 * w

print(f"  y = w^2 + 2w = {y.item()},  grad_fn: {y.grad_fn}")

y_detached = y.detach()
print(f"  y.detach():           {y_detached.item()},  "
      f"grad_fn: {y_detached.grad_fn},  "
      f"requires_grad: {y_detached.requires_grad}")

# Common pattern: log the loss value without keeping the graph
loss_value = y.detach().item()   # pure Python float, no graph
print(f"\n  Safe loss logging: loss = {loss_value}  (Python float, no graph)")

