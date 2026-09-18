# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# %% 

# Quantum circuit circulation


N_QUBITS = 4
N_LAYERS  = 3

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')

    # Variational layers
    for l in range(N_LAYERS):
        # Parameterised rotations
        for i in range(N_QUBITS):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)

        # Entanglement (CNOT chain)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])

    return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]



# %% 

class HybridIrisClassifier(nn.Module):
    def __init__(self, n_qubits=N_QUBITS, n_layers=N_LAYERS, n_classes=3):
        super().__init__()
        self.n_qubits = n_qubits

        # Classical encoder: 4 features → n_qubits rotation angles
        # Tanh squashes to (-1, 1) ≈ angle range for AngleEmbedding
        self.encoder = nn.Sequential(
            nn.Linear(4, n_qubits),
            nn.Tanh(),
        )

        # Quantum variational layer
        self.q_layer = qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes={"weights": (n_layers, n_qubits, 3)}
        )

        # Classical classification head
        # Reads ⟨Z⟩ outputs → class logits
        self.head = nn.Linear(n_qubits, n_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.encoder[0].weight)
        nn.init.zeros_(self.encoder[0].bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 4) float32
        returns: (batch, 3) logits
        """
        # Encode to qubit angles (float64 for PennyLane)
        x = self.encoder(x).double()

        # Quantum circuit: (batch, n_qubits) → (batch, n_qubits)
        x = self.q_layer(x)

        # Classify: (batch, n_qubits) → (batch, n_classes)
        x = self.head(x.float())

        return x   # raw logits — CrossEntropyLoss handles softmax

    def parameter_summary(self):
        """Print parameter breakdown."""
        classical = sum(p.numel() for name, p in self.named_parameters()
                        if 'weights' not in name)
        quantum   = sum(p.numel() for name, p in self.named_parameters()
                        if 'weights' in name)
        total     = classical + quantum
        print(f"\n  Parameter breakdown:")
        print(f"    Classical (encoder + head): {classical}")
        print(f"    Quantum  (VQC weights):     {quantum}")
        print(f"    Total:                      {total}")
        return {'classical': classical, 'quantum': quantum, 'total': total}


# %% 

model = HybridIrisClassifier()
model.parameter_summary()

# Test forward pass
x_test = torch.randn(5, 4)
with torch.no_grad():
    out = model(x_test)

print(f"\n  Input  shape: {x_test.shape}")
print(f"  Output shape: {out.shape}   (logits, 3 classes)")
print(f"  Predicted classes: {out.argmax(dim=1).tolist()}")

print("Quantum Circuit")
print(qml.draw(quantum_circuit)(
    torch.zeros(N_QUBITS, dtype=torch.float64),
    torch.zeros(N_LAYERS, N_QUBITS, 3, dtype=torch.float64)
))

specs = qml.specs(quantum_circuit)(
    torch.zeros(N_QUBITS, dtype=torch.float64),
    torch.zeros(N_LAYERS, N_QUBITS, 3, dtype=torch.float64)
)
print(f"\n  Circuit depth:  {specs['resources'].depth}")
print(f"  Gate types:     {dict(specs['resources'].gate_types)}")

# %% 