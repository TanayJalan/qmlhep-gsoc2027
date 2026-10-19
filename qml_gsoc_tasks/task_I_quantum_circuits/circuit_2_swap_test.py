
import pennylane as qml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(0)

# ── Device: 1 ancilla + 2 qubits for |ψ⟩ + 2 qubits for |φ⟩ ──
dev = qml.device("default.qubit", wires=5)


def state_zero(wires):
    """Prepare |00⟩ — default, no gates needed."""
    pass

def state_one(wires):
    """Prepare |11⟩."""
    qml.PauliX(wires=wires[0])
    qml.PauliX(wires=wires[1])

def state_plus(wires):
    """Prepare |++⟩ = H⊗H|00⟩."""
    qml.Hadamard(wires=wires[0])
    qml.Hadamard(wires=wires[1])

def state_bell(wires):
    """Prepare Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2."""
    qml.Hadamard(wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])

def state_rx(wires, theta=np.pi / 4):
    """Prepare Rx(theta)|00⟩ — partially rotated state."""
    qml.RX(theta, wires=wires[0])
    qml.RX(theta, wires=wires[1])

@qml.qnode(dev)
def swap_test(prepare_psi, prepare_phi):
    # Step 1: Prepare the two states to compare
    prepare_psi(wires=[1, 2])
    prepare_phi(wires=[3, 4])

    # Step 2: Hadamard on ancilla → superposition
    qml.Hadamard(wires=0)

    # Step 3: Controlled-SWAP (Fredkin) between registers
    # CSWAP(ancilla, psi_qubit, phi_qubit) for each pair
    qml.CSWAP(wires=[0, 1, 3])   # controlled swap of psi_0 and phi_0
    qml.CSWAP(wires=[0, 2, 4])   # controlled swap of psi_1 and phi_1

    # Step 4: Hadamard on ancilla
    qml.Hadamard(wires=0)

    # Step 5: Measure ancilla
    return qml.probs(wires=0)


def run_swap_test(prepare_psi, prepare_phi, label_psi, label_phi, expected_sim):
    """Run SWAP test and print formatted results."""
    probs      = swap_test(prepare_psi, prepare_phi)
    p0         = float(probs[0])
    similarity = max(0.0, 2 * p0 - 1)   # clip numerical noise at 0
    ok         = abs(similarity - expected_sim) < 1e-5
    return p0, similarity, ok

print("  Circuit diagram (example with |00⟩ vs |++⟩):")
print(qml.draw(swap_test)(state_zero, state_plus))


print("\n  Running SWAP test on multiple state pairs:")
print(f"\n  {'States':^32}  {'P(0)':>6}  {'|<ψ|φ>|²':>10}  "
      f"{'Expected':>10}  {'Status':>6}")
print(f"  {'-'*70}")

test_cases = [
    # (prep_psi, prep_phi, label_psi, label_phi, expected_similarity)
    (state_zero, state_zero,  "|00⟩",  "|00⟩",  1.0),
    (state_one,  state_one,   "|11⟩",  "|11⟩",  1.0),
    (state_zero, state_one,   "|00⟩",  "|11⟩",  0.0),
    (state_zero, state_plus,  "|00⟩",  "|++⟩",  0.25),
    (state_plus, state_plus,  "|++⟩",  "|++⟩",  1.0),
    (state_bell, state_bell,  "|Φ+⟩",  "|Φ+⟩",  1.0),
    (state_zero, state_bell,  "|00⟩",  "|Φ+⟩",  0.5),
    (state_plus, state_bell,  "|++⟩",  "|Φ+⟩",  0.5),
]

all_ok = True
results_table = []

for (pp, pf, lp, lf, exp_sim) in test_cases:
    p0, sim, ok = run_swap_test(pp, pf, lp, lf, exp_sim)
    all_ok = all_ok and ok
    label  = f"{lp} vs {lf}"
    status = "ok" if ok else "FAIL"
    results_table.append((label, p0, sim, exp_sim, ok))
    print(f"  {label:^32}  {p0:>6.4f}  {sim:>10.4f}  "
          f"{exp_sim:>10.4f}  {status:>6}")

print(f"\n  All tests passed: {'YES ok' if all_ok else 'NO -- check circuit'}")

print(f"""
  Formula verification:
      |⟨ψ|φ⟩|² = 2 × P(ancilla=0) − 1

      Case: |00⟩ vs |00⟩ (identical)
          P(0) = {float(swap_test(state_zero, state_zero)[0]):.6f}
          Similarity = 2 × {float(swap_test(state_zero, state_zero)[0]):.6f} − 1
                     = {2*float(swap_test(state_zero, state_zero)[0])-1:.6f}  (expected 1.0) ok

      Case: |00⟩ vs |11⟩ (orthogonal)
          P(0) = {float(swap_test(state_zero, state_one)[0]):.6f}
          Similarity = 2 × {float(swap_test(state_zero, state_one)[0]):.6f} − 1
                     = {2*float(swap_test(state_zero, state_one)[0])-1:.6f}  (expected 0.0) ok
""")

print("  Similarity vs rotation angle theta (|00⟩ vs Rx(theta)|00⟩):")
print(f"\n  {'theta/pi':>8}  {'P(0)':>8}  {'similarity':>12}  {'cos²(theta/2)':>14}")
print(f"  {'-'*48}")

thetas = np.linspace(0, np.pi, 9)
sims   = []

for theta in thetas:
    def prep_theta(wires, t=theta):
        qml.RX(t, wires=wires[0])
        qml.RX(t, wires=wires[1])

    p0     = float(swap_test(state_zero, prep_theta)[0])
    sim    = max(0.0, 2 * p0 - 1)
    theory = np.cos(theta / 2) ** 4   # for 2-qubit state
    sims.append(sim)
    print(f"  {theta/np.pi:>8.3f}  {p0:>8.4f}  {sim:>12.4f}  {theory:>14.4f}")


os.makedirs('figures', exist_ok=True)

# Circuit diagram
try:
    fig_circ, _ = qml.draw_mpl(swap_test, style='pennylane')(state_zero, state_plus)
    fig_circ.suptitle('Task I — Circuit 2: SWAP Test\n'
                       'Ancilla(0) + |ψ⟩(1,2) + |φ⟩(3,4)',
                       fontsize=10, y=1.05)
    fig_circ.tight_layout()
    fig_circ.savefig('figures/circuit_2.png', dpi=150, bbox_inches='tight')
    plt.close(fig_circ)
    print("\n  Saved: figures/circuit_2.png")
except Exception as e:
    print(f"\n  (circuit diagram: {e})")

# Similarity vs angle plot
fig_sim, ax = plt.subplots(figsize=(8, 4))
theta_dense = np.linspace(0, np.pi, 50)
sims_dense  = []
for theta in theta_dense:
    def prep_t(wires, t=float(theta)):
        qml.RX(t, wires=wires[0])
        qml.RX(t, wires=wires[1])
    p0 = float(swap_test(state_zero, prep_t)[0])
    sims_dense.append(max(0.0, 2 * p0 - 1))

ax.plot(theta_dense / np.pi, sims_dense, color='steelblue',
        linewidth=2, label='SWAP test similarity')
ax.plot(theta_dense / np.pi, np.cos(theta_dense / 2) ** 4,
        color='red', linestyle='--', linewidth=1.5, alpha=0.7,
        label='cos⁴(θ/2) theory')
ax.set_xlabel('θ / π')
ax.set_ylabel('|⟨ψ|φ⟩|²')
ax.set_title('Task I Circuit 2 — Similarity vs Rotation Angle')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig_sim.savefig('figures/circuit_2_similarity.png', dpi=150, bbox_inches='tight')
plt.close(fig_sim)
print("  Saved: figures/circuit_2_similarity.png")

