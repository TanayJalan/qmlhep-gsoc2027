import pennylane as qml
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

dev = qml.device("default.qubit", wires=5)

@qml.qnode(dev)
def circuit():
    # Layer 1: Hadamard superposition
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    # Layer 2: CNOT entangling chain
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    # Layer 3: SWAP
    qml.SWAP(wires=[1, 3])
    # Layer 4: Rx rotation
    qml.RX(np.pi / 3, wires=4)
    return [qml.expval(qml.PauliZ(w)) for w in range(5)]

@qml.qnode(dev)
def circuit_probs():
    qml.Hadamard(wires=0); qml.Hadamard(wires=1); qml.Hadamard(wires=2)
    qml.CNOT(wires=[0,1]); qml.CNOT(wires=[1,2])
    qml.CNOT(wires=[2,3]); qml.CNOT(wires=[3,4])
    qml.SWAP(wires=[1,3])
    qml.RX(np.pi/3, wires=4)
    return qml.probs(wires=range(5))

print("\n  Circuit diagram:")
print(qml.draw(circuit)())

specs = qml.specs(circuit)()
print(f"\n  Gate types:  {dict(specs['resources'].gate_types)}")
print(f"  Depth:       {specs['resources'].depth}")
print(f"  Total gates: {specs['resources'].num_gates}")

expvals = circuit()
print(f"\n  <Z> expectation values:")
for w, val in enumerate(expvals):
    print(f"    qubit {w}: <Z> = {float(val):+.8f}")

probs = circuit_probs()
print(f"\n  Top 8 basis states:")
print(f"  {'State':>8}  {'Prob':>10}  Bar")
print(f"  {'-'*40}")
top8 = np.argsort(probs)[-8:][::-1]
for idx in top8:
    p   = float(probs[idx])
    bar = 'X' * int(p * 30)
    print(f"  |{format(idx,'05b')}>  {p:>10.6f}  {bar}")

print(f"\n  Sum of probs: {float(probs.sum()):.10f}  ok")
print(f"\n  Note: <Z4> approx cos(pi/3) = {np.cos(np.pi/3):.4f} (modified by entanglement)")

# Save figures
os.makedirs('figures', exist_ok=True)
try:
    fig, _ = qml.draw_mpl(circuit, style='pennylane')()
    fig.suptitle('Task I - Circuit 1: 5-Qubit Circuit', fontsize=10)
    fig.tight_layout()
    fig.savefig('figures/circuit_1.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n  Saved: figures/circuit_1.png")
except Exception as e:
    print(f"\n  (circuit diagram: {e})")

fig2, ax = plt.subplots(figsize=(10, 4))
top16    = np.argsort(probs)[-16:][::-1]
ax.bar(range(16), [float(probs[i]) for i in top16], color='steelblue', alpha=0.8)
ax.set_xticks(range(16))
ax.set_xticklabels([f"|{format(i,'05b')}>" for i in top16], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Probability'); ax.set_title('Circuit 1 -- Top 16 State Probabilities')
ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
fig2.savefig('figures/circuit_1_probs.png', dpi=150, bbox_inches='tight'); plt.close(fig2)
print("  Saved: figures/circuit_1_probs.png")
