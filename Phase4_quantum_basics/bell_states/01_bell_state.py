import pennylane as qml
import numpy as np

np.random.seed(0)
dev2  = qml.device("default.qubit", wires=2)
root2 = 1.0 / np.sqrt(2)


@qml.qnode(dev2)
def phi_plus():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

@qml.qnode(dev2)
def phi_minus():
    qml.PauliX(wires=0)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

@qml.qnode(dev2)
def psi_plus():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=1)
    return qml.state()

@qml.qnode(dev2)
def psi_minus():
    qml.PauliX(wires=0)
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=1)
    return qml.state()

bell_states = {
    '|Phi+>': (phi_plus(),  [ root2,      0,      0,  root2]),
    '|Phi->': (phi_minus(), [-root2,      0,      0,  root2]),
    '|Psi+>': (psi_plus(),  [     0,  root2,  root2,      0]),
    '|Psi->': (psi_minus(), [     0, -root2,  root2,      0]),
}

print(f"\n {'State':>7}  {'|00>':>8}  {'|01>':>8}  {'|10>':>8}  {'|11>':>8}  Status")
print(f"  {'-'*58}")
for name, (state, expected) in bell_states.items():
    s  = np.round(state.real, 4)
    ok = np.allclose(state, expected, atol=1e-6)
    print(f"  {name:>7}  {s[0]:>8.4f}  {s[1]:>8.4f}  {s[2]:>8.4f}  {s[3]:>8.4f}  {'ok' if ok else 'FAIL'}")

states = [s for s, _ in bell_states.values()]
names  = list(bell_states.keys())
all_ok = True
print("\n Inner products (diagonal=1, off-diagonal=0):")
for i, (si, ni) in enumerate(zip(states, names)):
    for j, (sj, nj) in enumerate(zip(states, names)):
        ip = abs(np.dot(si.conj(), sj))
        expected_ip = 1.0 if i == j else 0.0
        ok = abs(ip - expected_ip) < 1e-9
        if not ok: all_ok = False

@qml.qnode(dev2)
def measure_phi_plus():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0]), qml.probs(wires=[1]), qml.probs(wires=[0,1])

p0, p1, p01 = measure_phi_plus()
print(f"\n |Phi+> = (|00>+|11>)/sqrt(2):")
print(f" Qubit 0 alone: P(0)={p0[0]:.3f} P(1)={p0[1]:.3f} (50/50)")
print(f" Qubit 1 alone: P(0)={p1[0]:.3f} P(1)={p1[1]:.3f} (50/50)")
print(f" Joint:  P(00)={p01[0]:.3f} P(01)={p01[1]:.3f} P(10)={p01[2]:.3f} P(11)={p01[3]:.3f}")
print(f" Individually random -- jointly perfectly correlated. That IS entanglement.")

print("\n  |Phi+>:"); print(qml.draw(phi_plus)())
print("\n  |Phi->:"); print(qml.draw(phi_minus)())
print("\n  |Psi+>:"); print(qml.draw(psi_plus)())
print("\n  |Psi->:"); print(qml.draw(psi_minus)())