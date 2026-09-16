# %%0

from numpy import float64
from pennylane.math import requires_grad
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"  PyTorch device: {DEVICE}")
print(f"  PennyLane version: {qml.__version__}")


# %%1

n_qubits = 2
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="autograd")
def circuit_autograd(weights, x):
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="torch")
def circuit_torch(weights, x):
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

import pennylane.numpy as pnp

w_pl = pnp.array([0.5, 0.3], requires_grad=True)
x_pl = pnp.array([0.1, 0.8])

w_pt = torch.tensor([0.5, 0.3], requires_grad=True, dtype=torch.float64)
x_pt = torch.tensor([0.1, 0.8], dtype=torch.float64)

out_autograd = circuit_autograd(w_pl, x_pl)
out_torch    = circuit_torch(w_pt, x_pt)

print(f"\n  autograd output: {float(out_autograd):.6f}  type: {type(out_autograd).__name__}")
print(f"  torch    output: {out_torch.item():.6f}  type: {type(out_torch).__name__}")
print(f"  Values match: {'ok' if abs(float(out_autograd) - out_torch.item()) < 1e-6 else 'FAIL'}")
print(f"\n  torch output has grad_fn: {out_torch.grad_fn is not None}")
print(f"  → can call .backward() on it directly")

# %%2

w = torch.tensor([0.5, 0.3], requires_grad=True, dtype=torch.float64)
x = torch.tensor([0.4, 0.7], dtype=torch.float64)
t = torch.tensor(1.0,        dtype=torch.float64)   # target

out  = circuit_torch(w, x)
loss = (out - t) ** 2
loss.backward()

print(f"\n  Weights:{w.tolist()}")
print(f"  Input x:{x.tolist()}")
print(f"  Output:{out.item():.6f}")
print(f"  Loss:{loss.item():.6f}")
print(f"  dL/dw:{w.grad.tolist()}")
print(f"  Gradient is a torch.Tensor: {isinstance(w.grad, torch.Tensor)}")


eps = 1e-5
with torch.no_grad():
    w_p = w.clone(); w_p[0] += eps
    w_m = w.clone(); w_m[0] -= eps
    fd  = ((circuit_torch(w_p, x) - t)**2 -
           (circuit_torch(w_m, x) - t)**2) / (2 * eps)

print(f"\n  Gradient check dL/dw[0]:")
print(f"PyTorch backward: {w.grad[0].item():.8f}")
print(f"Finite diff: {fd.item():.8f}")
print(f"Diff: {abs(w.grad[0].item() - fd.item()):.2e}  {'ok' if abs(w.grad[0].item() - fd.item()) < 1e-4 else 'FAIL'}")


# %%3


n_qubits = 2
n_layers  = 2
dev2 = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev2, interface="torch")
def deep_circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
        qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

# Target: learn to output +1 for class +1 data
X_pos = torch.tensor([[0.8, 0.9], [0.7, 0.85], [0.9, 0.8]], dtype=torch.float64)
X_neg = torch.tensor([[0.1, 0.2], [0.2, 0.1], [0.15, 0.25]], dtype=torch.float64)
X_all = torch.cat([X_pos, X_neg], dim=0)
Y_all = torch.tensor([1., 1., 1., -1., -1., -1.], dtype=torch.float64)

weights = torch.randn(n_layers, n_qubits, 2,
                       dtype=torch.float64, requires_grad=True)
opt     = torch.optim.Adam([weights], lr=0.1)

print(f"\n  Training QNode with Adam optimizer (20 steps):")
print(f"  {'Step':>5}  {'Loss':>10}  {'Acc':>8}")
print(f"  {'-'*28}")

for step in range(1, 21):
    opt.zero_grad()
    preds = torch.stack([deep_circuit(weights, x) for x in X_all])
    loss  = torch.mean((preds - Y_all) ** 2)
    loss.backward()
    opt.step()

    if step % 4 == 0 or step == 1:
        with torch.no_grad():
            acc = (torch.sign(preds) == Y_all).float().mean() * 100
        print(f"  {step:>5}  {loss.item():>10.6f}  {acc.item():>7.1f}%")

# %% 4

dev3 = qml.device("default.qubit", wires=3)

@qml.qnode(dev3, interface="torch")
def multi_output_circuit(weights, x):
    """Returns ⟨Z⟩ for each qubit — 3-element output vector."""
    qml.AngleEmbedding(x, wires=range(3), rotation='Y')
    for i in range(3):
        qml.RY(weights[i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    return [qml.expval(qml.PauliZ(w)) for w in range(3)]

w3 = torch.randn(3, dtype=torch.float64, requires_grad=True)
x3 = torch.tensor([0.3, 0.6, 0.9], dtype=torch.float64)

out3 = torch.stack(multi_output_circuit(w3, x3))
print(f"\n Output vector: {out3.tolist()}")
print(f" Shape:{out3.shape}")
print(f"Has grad_fn:{out3.grad_fn is not None}")

loss3 = out3.sum()
loss3.backward()
print(f"Gradient:{w3.grad.tolist()}")

# %%