
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

N_NODE_FEATURES = 5    # [delta_eta, delta_phi, log_pT, log_E, charge]
N_CLASSES       = 2    # 0 = QCD background, 1 = top quark signal


class JetGCN(nn.Module):
    def __init__(self,
                 in_channels: int = N_NODE_FEATURES,
                 hidden:      int = 64,
                 n_classes:   int = N_CLASSES,
                 dropout:     float = 0.3):
        super().__init__()

        # Message passing layers
        self.conv1 = GCNConv(in_channels, hidden)
        self.bn1   = nn.BatchNorm1d(hidden)

        self.conv2 = GCNConv(hidden, hidden)
        self.bn2   = nn.BatchNorm1d(hidden)

        self.conv3 = GCNConv(hidden, hidden)
        self.bn3   = nn.BatchNorm1d(hidden)

        # Classifier head (applied after global pooling)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Data) -> torch.Tensor:
        """
        data.x          : (N_total, in_channels)  node features
        data.edge_index : (2, E)                  graph connectivity
        data.batch      : (N_total,)               node → graph mapping

        Returns: (batch_size, n_classes) logits
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1: message passing + normalisation
        x = F.relu(self.bn1(self.conv1(x, edge_index)))

        # Layer 2
        x = F.relu(self.bn2(self.conv2(x, edge_index)))

        # Layer 3
        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        # Global pooling: (N_total, hidden) → (batch_size, hidden)
        x = global_mean_pool(x, batch)

        # Classify
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Sanity check ─────────────────────────────────────────────
if __name__ == "__main__":
    from torch_geometric.data import Batch

    # Dummy graph: 20 nodes, 160 edges (k=8 k-NN), batch size 4
    def dummy_graph():
        n = 20
        x  = torch.randn(n, N_NODE_FEATURES)
        ei = torch.randint(0, n, (2, n * 8))
        y  = torch.randint(0, 2, (1,))
        return Data(x=x, edge_index=ei, y=y, num_nodes=n)

    batch = Batch.from_data_list([dummy_graph() for _ in range(4)])
    model = JetGCN()

    print(f"JetGCN parameters: {model.count_parameters():,}")
    with torch.no_grad():
        out = model(batch)
    print(f"Input nodes: {batch.x.shape}  Output: {out.shape}")
    assert out.shape == (4, 2), f"Expected (4,2), got {out.shape}"
    print("Sanity check passed ok")
