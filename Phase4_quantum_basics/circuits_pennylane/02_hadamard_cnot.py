import pennylane as qml
import numpy as np

np.random.seed(0)


dev2 = qml.device("default.qubit", wires=2)
dev3 = qml.device("default.qubit", wires=3)

@qml.qnode(dev2)
def cnot_demo(c, t):
    if c == 1: qml.PauliX(wires=0)
    if t == 1: qml.PauliX(wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])

print("\n  CNOT truth table:")
inputs = [(0,0),(0,1),(1,0),(1,1)]
expected = ['|00>', '|01>', '|11>', '|10>']
for (c,t), exp in zip(inputs, expected):
    probs = cnot_demo(c, t)
    out_i = int(np.argmax(probs))
    out_s = f"|{out_i//2}{out_i%2}>"
    ok = out_s == exp
    print(f"|{c}{t}> -> {out_s} {'ok' if ok else 'FAIL'} (expected {exp})")

@qml.qnode(dev2)
def create_bell():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

@qml.qnode(dev2)
def bell_probs():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])

state = create_bell()
probs = bell_probs()
print(f"\n State vector: {np.round(state.real, 4)}")
print(f" Probs |00>={probs[0]:.4f} |01>={probs[1]:.4f} |10>={probs[2]:.4f} |11>={probs[3]:.4f}")
print(f" Only |00> and |11> -- qubits always agree when measured")
print(f"\n Circuit:")
print(qml.draw(create_bell)())


@qml.qnode(dev2)
def swap_demo():
    qml.PauliX(wires=0)
    qml.SWAP(wires=[0, 1])
    return qml.probs(wires=[0, 1])

p = swap_demo()
out_i = int(np.argmax(p))
print(f"Input |10>  After SWAP: |{out_i//2}{out_i%2}> (qubit states exchanged ok)")

@qml.qnode(dev3)
def toffoli_demo(c1, c2, t):
    if c1: qml.PauliX(wires=0)
    if c2: qml.PauliX(wires=1)
    if t:  qml.PauliX(wires=2)
    qml.Toffoli(wires=[0, 1, 2])
    return qml.probs(wires=[0, 1, 2])

print("\n Toffoli cases:")
for (c1, c2, t) in [(1,1,0),(1,1,1),(1,0,0),(0,1,0)]:
    p = toffoli_demo(c1, c2, t)
    out_i = int(np.argmax(p))
    print(f"|{c1}{c2}{t}> -> |{format(out_i,'03b')}>")

@qml.qnode(dev2)
def marginals():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0]), qml.probs(wires=[1]), qml.probs(wires=[0,1])

p0, p1, p01 = marginals()
print(f"\n Bell state |Phi+>:")
print(f"Qubit 0 marginal: {np.round(p0, 4)} (50/50 alone)")
print(f"Qubit 1 marginal: {np.round(p1, 4)} (50/50 alone)")
print(f"Joint probs: {np.round(p01, 4)} (correlated)")
print(f"Individually uniform but jointly entangled -- that IS entanglement.")

