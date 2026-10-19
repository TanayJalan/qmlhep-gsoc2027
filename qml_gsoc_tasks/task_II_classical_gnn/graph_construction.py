import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

np.random.seed(42)
torch.manual_seed(42)


PARTICLE_FEATURES = ['delta_eta', 'delta_phi', 'log_pT', 'log_E', 'charge']
EDGE_FEATURES     = ['delta_R', 'delta_eta', 'delta_phi']
N_NODE_FEATURES   = len(PARTICLE_FEATURES)
N_EDGE_FEATURES   = len(EDGE_FEATURES)


def generate_jet(n_constituents: int = 20,
                 is_top: bool = True,
                 rng=None) -> np.ndarray:
    """
    Generate one synthetic jet as a point cloud.

    Returns:
        array of shape (n_constituents, 5):
            [delta_eta, delta_phi, log_pT, log_E, charge]
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Top jets: wider spread, more sub-structure (3 sub-jets from t→Wbqq)
    # QCD jets: narrower, single core
    spread = 0.5 if is_top else 0.3

    pts       = rng.exponential(0.3, n_constituents)
    eta       = rng.normal(0, spread, n_constituents)
    phi       = rng.uniform(-np.pi, np.pi, n_constituents)
    E         = pts * np.cosh(eta)
    charges   = rng.choice([-1, 0, 1], n_constituents, p=[0.3, 0.4, 0.3])

    # Sub-jet structure for top quarks
    if is_top and n_constituents >= 6:
        for sub in range(3):
            center_eta = rng.normal(0, 0.3)
            center_phi = rng.uniform(-np.pi, np.pi)
            idx = slice(sub * (n_constituents // 3),
                        (sub + 1) * (n_constituents // 3))
            eta[idx] = rng.normal(center_eta, 0.1, len(eta[idx]))
            phi[idx] = rng.normal(center_phi, 0.1, len(phi[idx]))

    features = np.column_stack([
        eta,
        phi,
        np.log(np.clip(pts, 1e-6, None)),
        np.log(np.clip(E,   1e-6, None)),
        charges.astype(float),
    ])
    return features.astype(np.float32)


# Test
rng     = np.random.RandomState(42)
jet_top = generate_jet(20, is_top=True,  rng=rng)
jet_qcd = generate_jet(20, is_top=False, rng=rng)

print(f"\n  Top quark jet: {jet_top.shape}  (constituents × features)")
print(f"  QCD jet:       {jet_qcd.shape}")
print(f"\n  First 5 constituents of top jet:")
print(f"  {'deta':>8}  {'dphi':>8}  {'logpT':>8}  {'logE':>8}  {'q':>4}")
for row in jet_top[:5]:
    print(f"  {row[0]:>8.4f}  {row[1]:>8.4f}  {row[2]:>8.4f}  "
          f"{row[3]:>8.4f}  {row[4]:>4.0f}")



def build_knn_graph(features: np.ndarray,k: int = 8) -> Data:
    N = len(features)
    eta = features[:, 0]
    phi = features[:, 1]

    # Pairwise delta_R matrix — Phase 1 technique
    deta = eta[:, None] - eta[None, :]    # (N, N)
    dphi = phi[:, None] - phi[None, :]   # (N, N)

    # Wrap phi to [-pi, pi]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi

    dist = np.sqrt(deta ** 2 + dphi ** 2)   # (N, N)
    np.fill_diagonal(dist, np.inf)           # exclude self

    # For each node, find k nearest neighbours
    src_list, dst_list = [], []
    ea_list            = []

    for i in range(N):
        nn_idx = np.argsort(dist[i])[:k]    # k nearest
        for j in nn_idx:
            src_list.append(i)
            dst_list.append(int(j))
            ea_list.append([
                float(dist[i, j]),           # delta_R
                float(deta[i, j]),           # delta_eta
                float(dphi[i, j]),           # delta_phi
            ])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(ea_list, dtype=torch.float32)
    x          = torch.tensor(features, dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=N)


# Build graph
k     = 8
graph = build_knn_graph(jet_top, k=k)

print(f"\n  k-NN graph (k={k}) from top quark jet:")
print(f"    Nodes:       {graph.num_nodes}")
print(f"    Edges:       {graph.num_edges}  (= N × k = {graph.num_nodes*k})")
print(f"    Node feat:   {graph.x.shape}")
print(f"    Edge feat:   {graph.edge_attr.shape}")
print(f"    Edge index:  {graph.edge_index.shape}")

print(f"\n  First 5 edges (src → dst, delta_R):")
for i in range(5):
    s = graph.edge_index[0, i].item()
    d = graph.edge_index[1, i].item()
    r = graph.edge_attr[i, 0].item()
    print(f"    {s:>3} → {d:>3}   delta_R = {r:.4f}")

def build_radius_graph(features: np.ndarray, r: float = 0.4) -> Data:
    N   = len(features)
    eta = features[:, 0]
    phi = features[:, 1]

    deta = eta[:, None] - eta[None, :]
    dphi = phi[:, None] - phi[None, :]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    dist = np.sqrt(deta ** 2 + dphi ** 2)

    src_list, dst_list, ea_list = [], [], []
    for i in range(N):
        for j in range(N):
            if i != j and dist[i, j] < r:
                src_list.append(i)
                dst_list.append(j)
                ea_list.append([
                    float(dist[i, j]),
                    float(deta[i, j]),
                    float(dphi[i, j]),
                ])

    if not src_list:
        # No edges — return fully disconnected
        return Data(x=torch.tensor(features, dtype=torch.float32),
                    edge_index=torch.zeros(2, 0, dtype=torch.long),
                    edge_attr=torch.zeros(0, 3),
                    num_nodes=N)

    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor([src_list, dst_list], dtype=torch.long),
        edge_attr=torch.tensor(ea_list, dtype=torch.float32),
        num_nodes=N,
    )


r_graph = build_radius_graph(jet_top, r=0.4)
print(f"\n  Radius graph (r=0.4) from top quark jet:")
print(f"    Nodes:  {r_graph.num_nodes}")
print(f"    Edges:  {r_graph.num_edges}  (variable — depends on jet structure)")
print(f"    Avg degree: {r_graph.num_edges / r_graph.num_nodes:.2f}")

# Compare top vs QCD connectivity
g_top = build_radius_graph(jet_top, r=0.4)
g_qcd = build_radius_graph(jet_qcd, r=0.4)
print(f"\n  Top quark jet edges: {g_top.num_edges}  "
      f"(more spread → fewer close pairs)")
print(f"  QCD jet edges:       {g_qcd.num_edges}  "
      f"(more collimated → more close pairs)")


def jets_to_graphs(n_jets: int = 10, k: int = 8,
                    seed: int = 42) -> list:
    """Generate a list of PyG Data objects from synthetic jets."""
    rng_local = np.random.RandomState(seed)
    graphs = []
    for i in range(n_jets):
        is_top = (i % 2 == 0)
        feats  = generate_jet(20, is_top=is_top, rng=rng_local)
        graph  = build_knn_graph(feats, k=k)
        graph.y = torch.tensor([int(is_top)], dtype=torch.long)
        graphs.append(graph)
    return graphs


graphs = jets_to_graphs(n_jets=8, k=8)
batch  = Batch.from_data_list(graphs)

print(f"\n  Individual graphs: {len(graphs)}")
print(f"  After batching:")
print(f"    batch.x.shape:           {batch.x.shape}")
print(f"    batch.edge_index.shape:  {batch.edge_index.shape}")
print(f"    batch.edge_attr.shape:   {batch.edge_attr.shape}")
print(f"    batch.y.shape:           {batch.y.shape}")
print(f"    batch.batch.shape:       {batch.batch.shape}  "
      f"(node → graph mapping)")
print(f"    Unique graph indices:    {batch.batch.unique().tolist()}")

                     