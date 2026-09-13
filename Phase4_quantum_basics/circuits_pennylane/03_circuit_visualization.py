import pennylane as qml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)

dev3 = qml.device("default.qubit", wires=3)

@qml.qnode(dev3)
def example_circuit(theta):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(theta, wires=1)
    qml.CNOT(wires=[1, 2])
    qml.RZ(np.pi/4, wires=2)
    return [qml.expval(qml.PauliZ(w)) for w in range(3)]

print("\n  3-qubit example circuit:")
print(qml.draw(example_circuit)(theta=np.pi/3))

fig, ax = qml.draw_mpl(example_circuit, style='pennylane')(np.pi/4)
fig.suptitle('Example 3-Qubit Circuit', fontsize=11)
fig.tight_layout()
fig.savefig('example_circuit.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\n  Saved: example_circuit.png")

dev5 = qml.device("default.qubit", wires=5)

@qml.qnode(dev5)
def deep_circuit(theta):
    for i in range(5): qml.Hadamard(wires=i)
    for layer in range(3):
        for i in range(4): qml.CNOT(wires=[i, i+1])
        for i in range(5): qml.RY(theta * (layer+1), wires=i)
    return [qml.expval(qml.PauliZ(w)) for w in range(5)]

@qml.qnode(dev5)
def shallow_circuit(theta):
    for i in range(5): qml.RY(theta, wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[2, 3])
    return [qml.expval(qml.PauliZ(w)) for w in range(5)]

specs_d = qml.specs(deep_circuit)(theta=0.5)
specs_s = qml.specs(shallow_circuit)(theta=0.5)

print(f"\n Shallow: depth={specs_s['resources'].depth}  gates={specs_s['resources'].num_gates}")
print(f"Deep: depth={specs_d['resources'].depth}  gates={specs_d['resources'].num_gates}")
print(f"Deep gate types: {dict(specs_d['resources'].gate_types)}")
print(f"\nShallower = less noise on real quantum hardware.")

# Task I Circuit 1
@qml.qnode(dev5)
def task1_c1():
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.Hadamard(wires=2)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.SWAP(wires=[1, 3])
    qml.RX(np.pi/3, wires=4)
    return [qml.expval(qml.PauliZ(w)) for w in range(5)]

fig1, _ = qml.draw_mpl(task1_c1, style='pennylane')()
fig1.suptitle('Task I - Circuit 1: 5-Qubit Circuit', fontsize=10)
fig1.tight_layout()
fig1.savefig('task_I_circuit1.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("\n  Saved: task_I_circuit1.png")

# Task I Circuit 2
dev_sw = qml.device("default.qubit", wires=5)

@qml.qnode(dev_sw)
def task1_c2():
    qml.Hadamard(wires=1)
    qml.RY(np.pi/4, wires=3)
    qml.Hadamard(wires=0)
    qml.CSWAP(wires=[0, 1, 3])
    qml.CSWAP(wires=[0, 2, 4])
    qml.Hadamard(wires=0)
    return qml.probs(wires=0)

fig2, _ = qml.draw_mpl(task1_c2, style='pennylane')()
fig2.suptitle('Task I - Circuit 2: SWAP Test', fontsize=10)
fig2.tight_layout()
fig2.savefig('task_I_circuit2.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  Saved: task_I_circuit2.png")
print("\n  Copy both PNGs to: gsoc_tasks/task_I_quantum_circuits/figures/")
