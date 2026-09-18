"""
Phase 6 — Week 14
File: phase6_hep_papers/top_quark_baseline/data_loader.py

HEP data loader for top quark jet tagging.

The top quark tagging dataset (Kasieczka et al.) is the standard
benchmark used in QMLHEP Task II papers. Each sample is a jet
represented as a set of particle constituents.

Two modes:
    1. Real dataset  — download from Zenodo (1.2M jets, ~2GB)
    2. Synthetic     — generated locally for development/testing

The synthetic mode lets you develop and debug your entire pipeline
before committing to the large download. Shapes and feature
distributions match the real dataset closely enough for code testing.

Real dataset URL:
    https://zenodo.org/record/2603256
    Files: train.h5, val.h5, test.h5

Features (per jet constituent, up to 200 constituents):
    E, px, py, pz, eta, phi, pT, charge,
    is_electron, is_muon, is_photon,
    is_charged_hadron, is_neutral_hadron, is_EFlow_photon

For the GNN (Task II), we use the top-k constituents sorted by pT.

Run with:
    python data_loader.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import warnings

np.random.seed(42)
torch.manual_seed(42)


FEATURES = [
    'E', 'px', 'py', 'pz',           # 4-momentum
    'eta', 'phi', 'pT',               # kinematic
    'charge',                          # particle charge
    'is_electron', 'is_muon',          # lepton flags
    'is_photon',
    'is_charged_hadron',
    'is_neutral_hadron',
    'is_EFlow_photon',
]
N_FEATURES = len(FEATURES)             # 14

# Classes
CLASS_NAMES = ['QCD background', 'Top quark signal']

def generate_synthetic_jets(n_jets: int,
                              n_constituents: int = 30,
                              n_features: int = N_FEATURES,
                              signal_fraction: float = 0.5,
                              seed: int = 42) -> tuple:
    """
    Generate synthetic jet data that mimics the real top quark dataset.

    Real dataset characteristics reproduced:
        - Top jets have more constituents (heavier)
        - Top jet pT distribution is harder (higher energy)
        - Substructure: top jets have 3 sub-jets (from W→qq, b)
        - Continuous features follow approximately log-normal
        - Boolean features are sparse (mostly 0)

    Args:
        n_jets:           number of jet samples
        n_constituents:   particles per jet (padded)
        n_features:       features per constituent
        signal_fraction:  fraction of top quark jets
        seed:             random seed

    Returns:
        X: (n_jets, n_constituents, n_features) float32
        y: (n_jets,) int64 — 0=QCD, 1=top
    """
    rng = np.random.RandomState(seed)
    n_signal = int(n_jets * signal_fraction)
    n_bg     = n_jets - n_signal

    def make_jets(n, is_top):
        jets = np.zeros((n, n_constituents, n_features))

        # pT of leading constituent (higher for top)
        base_pt = rng.lognormal(mean=4.5 if is_top else 4.0,
                                 sigma=0.5, size=(n,))

        for i in range(n):
            # Number of active constituents (more for top)
            n_active = rng.randint(
                15 if is_top else 5,
                n_constituents + 1
            )
            n_active = min(n_active, n_constituents)

            # pT of each constituent (decreasing from leading)
            pts = base_pt[i] * np.sort(
                rng.exponential(0.3, n_active))[::-1]
            pts = np.clip(pts, 0.1, 1000)

            for j in range(n_active):
                pT  = pts[j]
                eta = rng.normal(0, 0.4 if is_top else 0.6)
                phi = rng.uniform(-np.pi, np.pi)

                # 4-momentum from pT, eta, phi (massless approx)
                px = pT * np.cos(phi)
                py = pT * np.sin(phi)
                pz = pT * np.sinh(eta)
                E  = pT * np.cosh(eta)

                jets[i, j, 0] = E
                jets[i, j, 1] = px
                jets[i, j, 2] = py
                jets[i, j, 3] = pz
                jets[i, j, 4] = eta
                jets[i, j, 5] = phi
                jets[i, j, 6] = pT

                # Charge: randomly +1, -1, or 0
                jets[i, j, 7] = rng.choice([-1, 0, 1],
                                             p=[0.3, 0.4, 0.3])

                # Particle type flags (sparse — mostly neutral hadrons)
                ptype = rng.choice(range(6),
                                    p=[0.05, 0.03, 0.08,
                                       0.42, 0.35, 0.07])
                if ptype < 6:
                    jets[i, j, 8 + ptype] = 1.0

        return jets

    X_sig = make_jets(n_signal, is_top=True)
    X_bg  = make_jets(n_bg,     is_top=False)
    y_sig = np.ones(n_signal,  dtype=np.int64)
    y_bg  = np.zeros(n_bg,     dtype=np.int64)

    X = np.concatenate([X_sig, X_bg], axis=0).astype(np.float32)
    y = np.concatenate([y_sig, y_bg], axis=0)

    # Shuffle
    idx = rng.permutation(n_jets)
    return X[idx], y[idx]

class JetDataset(Dataset):
    """
    PyTorch Dataset for jet classification.

    Supports two input modes:
        'graph'    : (n_constituents, n_features) per jet — for GNN (Task II)
        'sequence' : (n_constituents, n_features) per jet — for Transformer
        'flat'     : (n_constituents * n_features,) per jet — for MLP

    Args:
        X:       (N, n_const, n_features) numpy array
        y:       (N,) numpy array of labels
        mode:    'graph', 'sequence', or 'flat'
        top_k:   use only top-k constituents by pT (sorted)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 mode: str = 'sequence', top_k: int = 20,
                 pt_col: int = 6):
        self.mode  = mode
        self.top_k = top_k

        # Sort constituents by pT descending, take top_k
        sorted_idx = np.argsort(-X[:, :, pt_col], axis=1)
        X_sorted = X[np.arange(len(X))[:, None], sorted_idx, :]
        X = X_sorted[:, :top_k, :]

        if mode == 'flat':
            X = X.reshape(len(X), -1)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Real-data loader (Zenodo: https://zenodo.org/record/2603256) ──────────────
# Schema: 200 constituents × 4 features (E, PX, PY, PZ) stored flat as
#   columns E_0..E_199, PX_0..PX_199, PY_0..PY_199, PZ_0..PZ_199
# Label column: 'is_signal_new'  (0 = QCD background, 1 = top quark)
# We derive pT = sqrt(PX²+PY²) and eta = arcsinh(PZ/pT), phi = arctan2(PY,PX)
# giving 7 kinematics + 4 raw 4-momentum = 11 features (or keep 4 raw = faster).

N_CONSTITUENTS_REAL = 200          # full jet has up to 200 particles
N_FEATURES_REAL     = 4            # E, PX, PY, PZ per constituent


def load_real_jets(h5_path: str,
                   n_jets:  int = None,
                   start:   int = 0) -> tuple:
    """
    Load jets from Zenodo top-quark HDF5 file (PyTables/pandas format).

    Returns:
        X: (n_jets, N_CONSTITUENTS_REAL, 4)  float32   — (E, PX, PY, PZ)
        y: (n_jets,)                          int64
    """
    import pandas as pd
    stop = (start + n_jets) if n_jets is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.read_hdf(h5_path, key='table', start=start, stop=stop)

    n = len(df)
    y = df['is_signal_new'].values.astype(np.int64)

    # Reconstruct (n, 200, 4) from flat columns  E_0…E_199, PX_0…, PY_0…, PZ_0…
    E  = df[[f'E_{i}'  for i in range(N_CONSTITUENTS_REAL)]].values   # (n,200)
    PX = df[[f'PX_{i}' for i in range(N_CONSTITUENTS_REAL)]].values
    PY = df[[f'PY_{i}' for i in range(N_CONSTITUENTS_REAL)]].values
    PZ = df[[f'PZ_{i}' for i in range(N_CONSTITUENTS_REAL)]].values

    X = np.stack([E, PX, PY, PZ], axis=-1).astype(np.float32)  # (n,200,4)
    return X, y


def get_jet_dataloaders(n_train: int = 2000,
                         n_val:   int = 500,
                         n_test:  int = 500,
                         mode:    str = 'sequence',
                         top_k:   int = 20,
                         batch_size: int = 64,
                         data_dir: str = None) -> tuple:
    """
    Build train/val/test DataLoaders for jet classification.

    Args:
        data_dir: path to directory containing train.h5, val.h5, test.h5
                  (the real Zenodo dataset). If None, uses synthetic data.

    Returns:
        train_loader, val_loader, test_loader, sample_shape
    """
    if data_dir is not None:
        train_h5 = os.path.join(data_dir, 'train.h5')
        val_h5   = os.path.join(data_dir, 'val.h5')
        test_h5  = os.path.join(data_dir, 'test.h5')
        print(f'  [data_loader] Using REAL data from {data_dir}')
        X_train, y_train = load_real_jets(train_h5, n_jets=n_train)
        X_val,   y_val   = load_real_jets(val_h5,   n_jets=n_val)
        X_test,  y_test  = load_real_jets(test_h5,  n_jets=n_test)
        # pT column index for sorting in JetDataset: derive pT from PX, PY
        # We add pT as a 5th column so JetDataset can sort by it (col 4)
        pT_train = np.sqrt(X_train[..., 1]**2 + X_train[..., 2]**2)[..., None]
        pT_val   = np.sqrt(X_val[...,   1]**2 + X_val[...,   2]**2)[..., None]
        pT_test  = np.sqrt(X_test[...,  1]**2 + X_test[...,  2]**2)[..., None]
        X_train  = np.concatenate([X_train, pT_train], axis=-1)  # (n,200,5)
        X_val    = np.concatenate([X_val,   pT_val],   axis=-1)
        X_test   = np.concatenate([X_test,  pT_test],  axis=-1)
        n_cont   = 5   # all columns are continuous
    else:
        print('  [data_loader] Using SYNTHETIC data')
        X_train, y_train = generate_synthetic_jets(n_train, seed=42)
        X_val,   y_val   = generate_synthetic_jets(n_val,   seed=43)
        X_test,  y_test  = generate_synthetic_jets(n_test,  seed=44)
        n_cont = 7

    # Normalise continuous features using train stats
    orig_shape = X_train.shape
    X_flat = X_train.reshape(-1, orig_shape[-1])
    scaler = StandardScaler()
    scaler.fit(X_flat[:, :n_cont])

    def normalise(X):
        s = X.shape
        Xf = X.reshape(-1, s[-1]).copy()
        Xf[:, :n_cont] = scaler.transform(Xf[:, :n_cont])
        return Xf.reshape(s)

    X_train = normalise(X_train)
    X_val   = normalise(X_val)
    X_test  = normalise(X_test)

    # JetDataset sorts by pT col: col 6 for synthetic, col 4 for real (pT appended)
    pt_col = 4 if data_dir is not None else 6
    ds_train = JetDataset(X_train, y_train, mode=mode, top_k=top_k, pt_col=pt_col)
    ds_val   = JetDataset(X_val,   y_val,   mode=mode, top_k=top_k, pt_col=pt_col)
    ds_test  = JetDataset(X_test,  y_test,  mode=mode, top_k=top_k, pt_col=pt_col)

    kw = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(ds_train, batch_size=batch_size,
                               shuffle=True, **kw)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size,
                               shuffle=False, **kw)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size,
                               shuffle=False, **kw)

    sample_x, _ = ds_train[0]
    return train_loader, val_loader, test_loader, sample_x.shape

if __name__ == "__main__":

    print("Synthetic Jet Dataset")

    X, y = generate_synthetic_jets(n_jets=1000, n_constituents=30)

    print(f"\n  X shape: {X.shape}   (jets, constituents, features)")
    print(f"  y shape: {y.shape}")
    print(f"  Labels:  {np.bincount(y).tolist()}  (QCD, Top)")
    print(f"\n  Feature names: {FEATURES}")
    print(f"\n  Feature stats (first jet, first 5 constituents):")
    print(f"  {'Feature':<20}  {'Min':>8}  {'Max':>8}  {'Mean':>8}")
    print(f"  {'-'*50}")
    for i, fname in enumerate(FEATURES):
        vals = X[:, :, i].flatten()
        print(f"  {fname:<20}  {vals.min():>8.3f}  "
              f"{vals.max():>8.3f}  {vals.mean():>8.3f}")

    print("DataLoaders")

    train_l, val_l, test_l, sample_shape = get_jet_dataloaders(
        n_train=500, n_val=100, n_test=100,
        mode='sequence', top_k=20, batch_size=32
    )

    print(f"\n  Sample shape per jet: {sample_shape}  "
          f"(top_k={20}, n_features={N_FEATURES})")
    print(f"  Train batches: {len(train_l)}")
    print(f"  Val   batches: {len(val_l)}")
    print(f"  Test  batches: {len(test_l)}")

    xb, yb = next(iter(train_l))
    print(f"\n  First batch:")
    print(f"    X: {xb.shape}  y: {yb.shape}")
    print(f"    Label distribution: {yb.sum().item()} top / "
          f"{(yb==0).sum().item()} QCD")

    print("DONE — data_loader.py")
    print("""
  For real data:
    Download from https://zenodo.org/record/2603256
    Place train.h5, val.h5, test.h5 in phase6_hep_papers/data/
    Replace generate_synthetic_jets() with h5py loader
    """)
