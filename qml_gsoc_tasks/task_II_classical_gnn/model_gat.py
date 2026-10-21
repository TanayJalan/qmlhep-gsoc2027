import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

N_NODE_FEATURES = 5
N_CLASSES       = 2

class JetGAT(nn.Module):
    def __init__(self,
                 in_channels: int   = N_NODE_FEATURES,
                 hidden:      int   = 32,
                 heads:       int   = 4,
                 n_classes:   int   = N_CLASSES,
                 dropout:     float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.heads  = heads

        # Layer 1: concat=True → output dim = hidden * heads
        self.conv1 = GATConv(in_channels,    hidden, heads=heads,
                              dropout=dropout, concat=True)
        self.bn1   = nn.BatchNorm1d(hidden * heads)

        # Layer 2: concat=True
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads,
                              dropout=dropout, concat=True)
        self.bn2   = nn.BatchNorm1d(hidden * heads)

        # Layer 3: concat=False → output dim = hidden (single head mean)
        self.conv3 = GATConv(hidden * heads, hidden * 2, heads=1,
                              dropout=dropout, concat=False)
        self.bn3   = nn.BatchNorm1d(hidden * 2)

        # Mean + Max pool concatenated → 2 * (hidden*2) = hidden*4
        pool_out = hidden * 2 * 2
        self.head = nn.Sequential(
            nn.Linear(pool_out, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden * 2, n_classes),
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
        Returns: (batch_size, n_classes) logits
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer 1: 5 → 32*4=128 per node
        x = F.elu(self.bn1(self.conv1(x, edge_index)))

        # Layer 2: 128 → 128 per node
        x = F.elu(self.bn2(self.conv2(x, edge_index)))

        # Layer 3: 128 → 64 per node (single head)
        x = F.elu(self.bn3(self.conv3(x, edge_index)))

        # Mean + Max pool: (N_total, 64) → (batch, 128)
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x,  batch)
        x      = torch.cat([x_mean, x_max], dim=1)

        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ── Sanity check ─────────────────────────────────────────────
if __name__ == "__main__":
    from torch_geometric.data import Batch

    def dummy_graph():
        n  = 20
        x  = torch.randn(n, N_NODE_FEATURES)
        ei = torch.randint(0, n, (2, n * 8))
        y  = torch.randint(0, 2, (1,))
        return Data(x=x, edge_index=ei, y=y, num_nodes=n)

    batch = Batch.from_data_list([dummy_graph() for _ in range(4)])
    model = JetGAT()

    print(f"JetGAT parameters: {model.count_parameters():,}")
    with torch.no_grad():
        out = model(batch)
    print(f"Input nodes: {batch.x.shape}  Output: {out.shape}")
    assert out.shape == (4, 2)
    print("Sanity check passed ok")
