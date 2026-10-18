import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pennylane as qml
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math
import time
import multiprocessing

torch.manual_seed(42)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"  Device: {DEVICE}")
print(f"  PennyLane: {qml.__version__}")


N_QUBITS  = 4     # number of qubits per attention head
N_QLAYERS = 2     # variational circuit layers (re-upload between each)

dev_attn = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev_attn, interface="torch")
def quantum_attention_circuit(inputs, weights):
    """
    Data re-uploading variational circuit for one attention head.

    Args:
        inputs  : (N_QUBITS,) token embedding (truncated/projected)
        weights : (N_QLAYERS, N_QUBITS) rotation angles

    Returns:
        List of ⟨Z⟩ for each qubit → (N_QUBITS,) output
    """
    for l in range(N_QLAYERS):
        # Re-upload data at each layer (key innovation from 2209.08167)
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')

        # Variational rotations
        for i in range(N_QUBITS):
            qml.RY(weights[l, i], wires=i)

        # Entanglement
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        # Ring entanglement (last qubit back to first)
        if N_QUBITS > 2:
            qml.CNOT(wires=[N_QUBITS - 1, 0])

    return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]


class QuantumAttentionHead(nn.Module):
    """
    Single quantum attention head.

    Replaces the W_Q (or W_K, W_V) linear projection
    with a parameterised quantum circuit.

    Input:  (batch, seq, d_model)
    Output: (batch, seq, N_QUBITS)   — ⟨Z⟩ per qubit per token
    """
    def __init__(self, d_model: int, n_qubits: int = N_QUBITS,
                 n_layers: int = N_QLAYERS):
        super().__init__()
        self.d_model  = d_model
        self.n_qubits = n_qubits

        # Classical pre-projection: d_model → n_qubits
        # Maps token embedding to circuit input angles
        self.pre_proj = nn.Linear(d_model, n_qubits)

        # Quantum variational layer
        self.q_layer = qml.qnn.TorchLayer(
            quantum_attention_circuit,
            weight_shapes={"weights": (n_layers, n_qubits)}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq, d_model)
        returns: (batch, seq, n_qubits)
        """
        B, S, _ = x.shape

        # Project to qubit dimension
        x_proj = torch.tanh(self.pre_proj(x))        # (B, S, n_qubits)

        # Apply quantum circuit to each token independently
        # Flatten batch and sequence dimensions for TorchLayer
        x_flat = x_proj.reshape(B * S, self.n_qubits).double()
        q_out  = self.q_layer(x_flat)                # (B*S, n_qubits)
        return q_out.float().reshape(B, S, self.n_qubits)


# Test quantum attention head
if __name__ == "__main__":
    head = QuantumAttentionHead(d_model=64, n_qubits=N_QUBITS)
    x_test = torch.randn(2, 5, 64)   # 2 samples, 5 tokens, 64-dim embedding

    with torch.no_grad():
        out = head(x_test)

    print(f"\n  Input  shape: {x_test.shape}")
    print(f"  Output shape: {out.shape}  (batch, seq, n_qubits)")
    print(f"  Output range: [{out.min().item():.3f}, {out.max().item():.3f}]  (⟨Z⟩ ∈ (-1,1))")

    head_params = sum(p.numel() for p in head.parameters())
    q_params    = sum(p.numel() for n, p in head.named_parameters()
                      if 'weights' in n)
    print(f"\n  Head parameters:     {head_params}")
    print(f"    Classical (proj):   {head_params - q_params}")
    print(f"    Quantum (circuit):  {q_params}")

    print("\n  Circuit diagram:")
    print(qml.draw(quantum_attention_circuit)(
        torch.zeros(N_QUBITS, dtype=torch.float64),
        torch.zeros(N_QLAYERS, N_QUBITS, dtype=torch.float64)
    ))

