# %% 0
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

# %% 1


n_qubits = 2
n_layers  = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_layer_fn(inputs, weights):
    """
    QNode to be wrapped.

    Args:
        inputs  : (n_qubits,) input features — NOT a trainable param
        weights : (n_layers, n_qubits, 3) — trainable gate angles
    """
    # Encode input
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')

    # Variational layers
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}

# Wrap as TorchLayer
qlayer = qml.qnn.TorchLayer(quantum_layer_fn, weight_shapes)

print(f"\n  TorchLayer created: {qlayer}")
print(f"\n  Learnable parameters:")
for name, param in qlayer.named_parameters():
    print(f"    {name}: shape={tuple(param.shape)}  numel={param.numel()}")

total = sum(p.numel() for p in qlayer.parameters())
print(f"\n  Total parameters: {total}  ({n_layers} × {n_qubits} × 3)")


# %%

x_single = torch.randn(n_qubits, dtype=torch.float64)
out_single = qlayer(x_single)
print(f"\n  Single sample:")
print(f"    Input  shape: {x_single.shape}")
print(f"    Output shape: {out_single.shape}")
print(f"    Output:       {[round(v, 4) for v in out_single.tolist()]}")

# Batch
batch = 4
x_batch = torch.randn(batch, n_qubits, dtype=torch.float64)
out_batch = qlayer(x_batch)
print(f"\n  Batch (size={batch}):")
print(f"    Input  shape: {x_batch.shape}")
print(f"    Output shape: {out_batch.shape}")
print(f"    Output:\n{out_batch.round(decimals=4)}")

hybrid_model = nn.Sequential(
    nn.Linear(4, n_qubits).double(),         # pre-process: 4→2
    qlayer,                                   # quantum layer: 2→2
    nn.Linear(n_qubits, 1).double(),          # classifier head: 2→1
    nn.Sigmoid(),
)

x_in = torch.randn(8, 4, dtype=torch.float64)
out  = hybrid_model(x_in)

print(f"\n  Model:")
for i, layer in enumerate(hybrid_model):
    print(f"    [{i}] {layer.__class__.__name__}")

print(f"\n  Input shape:  {x_in.shape}  (8 samples, 4 features)")
print(f"  Output shape: {out.shape}  (8 samples, 1 probability)")
print(f"  Output range: [{out.min().item():.4f}, {out.max().item():.4f}]  (sigmoid → (0,1))")

total_params = sum(p.numel() for p in hybrid_model.parameters())
print(f"\n  Total model parameters: {total_params}")
print(f"    Classical: {sum(p.numel() for name, p in hybrid_model.named_parameters() if 'weights' not in name)}")
print(f"    Quantum:   {sum(p.numel() for name, p in hybrid_model.named_parameters() if 'weights' in name)}")

# %%

class QuantumClassifier(nn.Module):
    """
    Hybrid quantum-classical binary classifier.
    Classical encoder → Quantum circuit → Classical head.
    """
    def __init__(self, in_features: int, n_qubits: int, n_layers: int):
        super().__init__()

        # Classical pre-processing
        self.encoder = nn.Sequential(
            nn.Linear(in_features, n_qubits),
            nn.Tanh(),
        )

        # Quantum variational layer
        dev_q = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev_q, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            for l in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(weights[l, i, 0], wires=i)
                    qml.RY(weights[l, i, 1], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.q_layer = qml.qnn.TorchLayer(
            circuit, {"weights": (n_layers, n_qubits, 3)}
        )

        # Classical output head
        self.head = nn.Linear(n_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x).double()    # classical encode → doubles for PL
        x = self.q_layer(x)             # quantum circuit
        x = self.head(x.float())        # classical head
        return torch.sigmoid(x)


model = QuantumClassifier(in_features=4, n_qubits=2, n_layers=2)
x_test = torch.randn(5, 4)
out    = model(x_test)

print(f"\n  QuantumClassifier:")
print(f"    Encoder:  Linear(4→2) + Tanh")
print(f"    Q-Layer:  TorchLayer(2 qubits, 2 layers)")
print(f"    Head:     Linear(2→1) + Sigmoid")
print(f"\n  Input:  {x_test.shape}  Output: {out.shape}")
print(f"  Output: {out.squeeze().tolist()}")
print(f"  Total params: {sum(p.numel() for p in model.parameters())}")

# %%


import tempfile, os

save_path = os.path.join(tempfile.gettempdir(), "quantum_model.pt")
torch.save(model.state_dict(), save_path)
print(f"\n  Saved to: {save_path}")
print(f"  State dict keys: {list(model.state_dict().keys())}")

# Load into fresh model
model2 = QuantumClassifier(in_features=4, n_qubits=2, n_layers=2)
model2.load_state_dict(torch.load(save_path, map_location='cpu'))

out2 = model2(x_test)
match = torch.allclose(out.detach(), out2.detach(), atol=1e-5)
print(f"  Loaded model output matches: {'ok' if match else 'FAIL'}")

# %%
