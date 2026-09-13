import pennylane as qml
import numpy as np

print(f"  PennyLane version: {qml.__version__}\n")

# default.qubit: classical simulation of quantum circuits
# Exact statevector simulation — no noise, no shot limit by default
dev_1q = qml.device("default.qubit", wires=1)   # 1-qubit device
dev_2q = qml.device("default.qubit", wires=2)   # 2-qubit device
dev_5q = qml.device("default.qubit", wires=5)   # 5-qubit device

# Section1 - QNode basics
@qml.qnode(dev_1q)
def identity_circuit():
    return qml.state()

state = identity_circuit()
print(f" |0) state vector: {state}")
print(f" (alpha = 1 for |0), beta = 0 for |1) -> [[alpha], [beta]]")

@qml.qnode(dev_1q)
def measure_z():
    return qml.expval(qml.PauliZ(0))

print(f"\n (z) on |0) : {measure_z()}")

print(f" Expected: +1.0 (|+1 eigenstate of z)")

# Pauli gates

@qml.qnode(dev_1q)
def x_gate():
    qml.PauliX(wires=0)           # flip |0⟩ to |1⟩
    return qml.state()

@qml.qnode(dev_1q)
def x_gate_expval():
    qml.PauliX(wires=0)
    return qml.expval(qml.PauliZ(0))

state_after_x = x_gate()
print(f"\n After x gate on |0)")
print(f" state vector: {state_after_x}  (now |1)")
print(f" (Z) =  {x_gate_expval():.1f}  (should be -1)")


@qml.qnode(dev_1q)
def z_gate():
    qml.PauliZ(wires=0)           # Z|0⟩ = |0⟩, no change in probabilities
    return qml.expval(qml.PauliZ(0))
print(f"\n After z gate on |0): (Z) = {z_gate():.1f} (Z|0> = |0), unchanged)")

@qml.qnode(dev_1q)
def y_gate():
    qml.PauliY(wires=0)
    return qml.probs(wires=0)

probs_y = y_gate()
print(f"\n After y gate on |0⟩: probs = {probs_y} (P(0)={probs_y[0]:.0f}, P(1)={probs_y[1]:.0f})")  


@qml.qnode(dev_1q)
def hadamard():
    qml.Hadamard(wires=0)
    return qml.state()

@qml.qnode(dev_1q)
def hadamard_probs():
    qml.Hadamard(wires=0)
    return qml.probs(wires=0)

@qml.qnode(dev_1q)
def hadamard_expval():
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(0))

state_h = hadamard()
probs_h = hadamard_probs()

print(f"\n H|0) state vector: {np.round(state_h, 4)}")
print(f"       = [1/sqrt(2), 1/sqrt(2)] = {[round(1/np.sqrt(2),4)]*2}")

print(f"Probabilities: P(0)={probs_h[0]:.4f} P(1)={probs_h[1]:.4f} (both 50%)")
print(f"⟨Z⟩ = {hadamard_expval():.6f} (should be ≈ 0 — equally likely 0 and 1)")

@qml.qnode(dev_1q)
def hadamard_twice():
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=0)         # H @ H = I
    return qml.state()


print(f"\n  H @ H |0⟩ = {hadamard_twice()}  (back to |0⟩ correct)")

# Rotation gates

@qml.qnode(dev_1q)
def rx_gate(theta):
    qml.RX(theta, wires=0)
    return qml.expval(qml.PauliZ(0))

angles = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
print(f"\n  Rx(θ) gate — ⟨Z⟩ vs angle θ:")
print(f"  {'θ/π':>6}  {'⟨Z⟩':>8}  {'cos(θ)':>8}  {'Match':>6}")
print(f"  {'-'*36}")
for theta in angles:
    z    = rx_gate(theta)
    cosv = np.cos(theta)
    ok   = abs(z - cosv) < 1e-6
    print(f"  {theta/np.pi:>6.2f}  {z:>8.4f}  {cosv:>8.4f}  {'✓' if ok else '✗':>6}")


print(f"\n⟨Z⟩ = cos(θ) for Rx — this is the parameter shift rule target.")

@qml.qnode(dev_1q)
def ry_gate(theta):
    qml.RY(theta, wires=0)
    return qml.probs(wires=0)

print(f"\nRy(θ) gate P(|0⟩) and P(|1⟩) vs angle:")
for theta in [0, np.pi/4, np.pi/2, np.pi]:
    p = ry_gate(theta)
    print(f"θ={theta/np.pi:.2f}π: P(0)={p[0]:.4f} P(1)={p[1]:.4f}")



@qml.qnode(dev_1q)
def s_gate_demo():
    qml.Hadamard(wires=0)     # |0⟩ → |+⟩
    qml.S(wires=0)            # |+⟩ → |+i⟩  (rotates phase)
    return qml.expval(qml.PauliY(0))

print(f"\n H then S on |0⟩:")
print(f"⟨Y⟩ = {s_gate_demo():.4f} (should be +1, state is |+i⟩ = Y eigenstate)")


@qml.qnode(dev_1q)
def example_circuit(theta):
    qml.Hadamard(wires=0)
    qml.RZ(np.pi / 4, wires=0)
    qml.RX(theta, wires=0)
    qml.S(wires=0)
    return qml.expval(qml.PauliZ(0))

print("\n Circuit diagram:")
print(qml.draw(example_circuit)(theta=np.pi/3))

print(f"\n Output ⟨Z⟩ = {example_circuit(np.pi/3):.4f}")

