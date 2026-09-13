import pennylane as qml
from pennylane import numpy as np
import numpy as _np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_np.random.seed(42)

n_qubits, n_layers = 2, 3
dev = qml.device("default.qubit", wires=n_qubits)

def vqc_layer(weights, wires):
    for i, wire in enumerate(wires):
        qml.RX(weights[i, 0], wires=wire)
        qml.RY(weights[i, 1], wires=wire)
        qml.RZ(weights[i, 2], wires=wire)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i+1]])

@qml.qnode(dev, interface="autograd")
def vqc(weights, x):
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
    for l in range(n_layers):
        vqc_layer(weights[l], wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

weights_init = np.array(
    _np.random.uniform(-_np.pi, _np.pi, (n_layers, n_qubits, 3)),
    requires_grad=True
)
x_sample = _np.array([0.5, 0.8])


print(f"\n Parameters: {weights_init.size}({n_layers} x {n_qubits} x 3)")
print(f" Circuit:")
print(qml.draw(vqc)(weights_init, x_sample))
print(f"Output for x={x_sample}: {float(vqc(weights_init, x_sample)):.6f}")

grad_fn = qml.grad(vqc, argnums=0)
grads = grad_fn(weights_init, x_sample)

print(f"\n Gradient shape: {grads.shape}")
print(f" Max |grad|: {_np.abs(grads).max():.6f}")

# Verify one element via finite difference
eps = 1e-5
w = weights_init.copy()
w_p = w.copy(); w_p[0,0,0]  += eps
w_m = w.copy(); w_m[0,0,0]  -= eps
fd = (float(vqc(w_p, x_sample)) - float(vqc(w_m, x_sample))) / (2*eps)
an = float(grads[0,0,0])
print(f"Gradient check weights[0,0,0]:")
print(f" Parameter shift: {an:.8f}")
print(f"Finite diff: {fd:.8f}")
print(f"Diff: {abs(an-fd):.2e} {'ok' if abs(an-fd) < 1e-4 else 'FAIL'}")

# Training

X_data = _np.array([
    [0.10, 0.20],[0.15, 0.25],[0.05, 0.30],[0.20, 0.10],
    [0.80, 0.90],[0.75, 0.85],[0.90, 0.80],[0.85, 0.95],
])
Y_data = _np.array([-1,-1,-1,-1,1,1,1,1], dtype=float)

weights = np.array(
    _np.random.uniform(-_np.pi, _np.pi, (n_layers, n_qubits, 3)),
    requires_grad=True
)

def cost(w, X, Y):
    preds = np.stack([vqc(w, x) for x in X])
    return np.mean((preds - Y)**2)

opt     = qml.GradientDescentOptimizer(stepsize=0.4)
n_epochs = 40
history  = []

print(f"\n Training {n_qubits}-qubit {n_layers}-layer VQC for {n_epochs} epochs...")
print(f"\n {'Epoch':>6}  {'Loss':>10}  {'Acc':>8}")
print(f"{'-'*30}")

for epoch in range(1, n_epochs + 1):
    weights, loss_val = opt.step_and_cost(
        lambda w: cost(w, X_data, Y_data), weights)
    history.append(float(loss_val))
    preds = _np.array([float(vqc(weights, x)) for x in X_data])
    acc = _np.mean(_np.sign(preds) == Y_data) * 100
    if epoch % 8 == 0 or epoch == 1:
        print(f" {epoch:>6} {float(loss_val):>10.6f}  {acc:>7.1f}%")


converged = history[-1] < history[0] * 0.5
print(f"\n Initial loss: {history[0]:.6f}")
print(f" Final loss: {history[-1]:.6f}")
print(f"Reduction: {(1 - history[-1]/history[0])*100:.1f}%")
print(f"\n {'PHASE 4 MILESTONE ok -- VQC converged' if converged else 'Try more epochs'}")

# Save plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history, color='steelblue', linewidth=2)
ax.axhline(history[0]*0.5, color='red', linestyle='--', alpha=0.6, label='50% threshold')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.set_title('VQC Training -- Phase 4 Milestone')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('vqc_training_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Loss curve saved: vqc_training_loss.png")
