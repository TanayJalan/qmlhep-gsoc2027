# QMLHEP GSoC 2027 — Implementation Plan

> **Project:** Quantum Transformer for High Energy Physics (QMLHEP5)
> **Organization:** ML4Sci
> **Target:** Google Summer of Code 2027
> **Current Phase:** Phase 1 — Python & NumPy Foundations

---

## Repository Folder Structure

```
QMLHEP-GSoC2027/
│
├── README.md                          # Project overview, progress tracker
├── requirements.txt                   # pip dependencies
├── environment.yml                    # conda environment
├── .gitignore                         # Python, Jupyter, macOS ignores
│
├── phase1_foundations/                # Weeks 1–2: Python & NumPy
│   ├── README.md
│   ├── numpy_basics/
│   │   ├── 01_array_ops.py
│   │   ├── 02_vectorization.py
│   │   ├── 03_broadcasting.py
│   │   └── 04_linear_algebra.py
│   ├── oop_patterns/
│   │   ├── 01_classes_and_modules.py
│   │   ├── 02_inheritance.py
│   │   └── 03_custom_iterator.py
│   └── backprop_from_scratch/
│       ├── 01_forward_pass.py
│       ├── 02_backward_pass.py
│       ├── 03_gradient_descent.py
│       └── 04_two_layer_net.py       # PHASE 1 MILESTONE
│
├── phase2_pytorch/                    # Weeks 3–5: Deep Learning
│   ├── README.md
│   ├── tensors_autograd/
│   │   ├── 01_tensors.py
│   │   ├── 02_autograd.py
│   │   └── 03_custom_backward.py
│   ├── training_loops/
│   │   ├── 01_basic_loop.py
│   │   └── 02_dataset_dataloader.py
│   └── cnn_cifar10/
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       └── results/                   # PHASE 2 MILESTONE: >80% accuracy
│
├── phase3_transformers/               # Weeks 6–8: Transformer Architecture
│   ├── README.md
│   ├── attention_from_scratch/
│   │   ├── 01_scaled_dot_product.py
│   │   ├── 02_multihead_attention.py
│   │   └── 03_positional_encoding.py
│   └── mini_vit_mnist/
│       ├── patch_embedding.py
│       ├── transformer_block.py
│       ├── vit_model.py
│       ├── train.py
│       └── results/                   # PHASE 3 MILESTONE + TASK VIII (part 1)
│
├── phase4_quantum_basics/             # Weeks 9–10: Quantum Computing
│   ├── README.md
│   ├── circuits_pennylane/
│   │   ├── 01_basic_gates.py
│   │   ├── 02_hadamard_cnot.py
│   │   └── 03_circuit_visualization.py
│   ├── bell_states/
│   │   └── 01_bell_state.py
│   └── variational_circuits/
│       └── 01_vqc_basics.py           # PHASE 4 MILESTONE
│
├── phase5_hybrid_qml/                 # Weeks 11–13: PennyLane + PyTorch
│   ├── README.md
│   ├── torch_layer/
│   │   ├── 01_qnode_basics.py
│   │   └── 02_torch_layer_intro.py
│   └── iris_classifier/
│       ├── classical_baseline.py
│       ├── hybrid_model.py
│       └── train.py                   # PHASE 5 MILESTONE
│
├── phase6_hep_papers/                 # Weeks 14–15: HEP Context
│   ├── README.md
│   ├── paper_notes/
│   │   ├── qvit_2110_06510.md         # arXiv:2110.06510 notes
│   │   └── qpt_2209_08167.md          # arXiv:2209.08167 notes
│   └── top_quark_baseline/
│       ├── data_loader.py
│       ├── classical_transformer.py
│       └── results/                   # PHASE 6 MILESTONE
│
├── phase7_qvit_prototype/             # Weeks 16–18: Build QVIT
│   ├── README.md
│   ├── quantum_attention.py
│   ├── qvit_model.py
│   ├── train.py
│   └── results/
│
├── gsoc_tasks/                        # Official QMLHEP evaluation tasks
│   ├── README.md
│   │
│   ├── task_I_quantum_circuits/       # MANDATORY
│   │   ├── README.md
│   │   ├── circuit_1_five_qubit.py    # 5-qubit: H + CNOT + SWAP + Rx
│   │   ├── circuit_2_swap_test.py     # SWAP test between |q1q2> and |q3q4>
│   │   └── figures/
│   │       ├── circuit_1.png
│   │       └── circuit_2.png
│   │
│   ├── task_II_classical_gnn/         # MANDATORY
│   │   ├── README.md
│   │   ├── data/
│   │   ├── graph_construction.py      # Point-cloud to graph
│   │   ├── model_gcn.py               # Architecture 1
│   │   ├── model_gat.py               # Architecture 2
│   │   ├── train.py
│   │   └── results/
│   │
│   ├── task_III_commentary/           # MANDATORY
│   │   └── commentary.md             # Original QC/QML commentary
│   │
│   └── task_VIII_vit_qvit/            # QMLHEP5 SPECIFIC
│       ├── README.md
│       ├── classical_vit/
│       │   ├── model.py
│       │   ├── train.py
│       │   └── results/               # MNIST accuracy report
│       └── quantum_vit_design/
│           └── architecture_sketch.md # QVIT design document
│
├── docs/
│   ├── weekly_log.md                  # What you did each week
│   ├── resources.md                   # All links, papers, courses
│   └── mentor_contact_log.md          # Track all interactions with ML4Sci
│
└── notebooks/                         # Experimental Jupyter notebooks
    └── scratch/                       # Throwaway exploration
```

---

## Phase 1 — Python & NumPy Deep Dive

**Duration:** Weeks 1–2
**Time commitment:** ~8 hours/week (16 hours total)
**Goal:** Build the mathematical and programming foundations every ML codebase assumes you already have.

---

### Week 1 — NumPy Mastery

#### Day 1–2: Array operations & vectorization

What to build in `phase1_foundations/numpy_basics/01_array_ops.py`:

```python
# Cover these concepts with working examples:
# - Array creation: np.zeros, np.ones, np.random.randn, np.arange
# - Indexing and slicing (1D, 2D, 3D)
# - Reshaping: reshape, flatten, squeeze, expand_dims
# - Axis operations: sum, mean, max along axes
# - Broadcasting rules (write 3 examples that would fail without it)
```

What to build in `02_vectorization.py`:

```python
# Write the same operation 3 ways and time them:
# 1. Pure Python loop
# 2. NumPy vectorized
# 3. NumPy with broadcasting
# Print speedup factors — this will be memorable
```

**Commit message:** `feat: numpy array ops and vectorization examples with timing comparison`

---

#### Day 3–4: Broadcasting & linear algebra

What to build in `03_broadcasting.py`:

```python
# Implement these from scratch using only NumPy:
# - Batch matrix multiply (without np.matmul)
# - Normalize a batch of vectors
# - Compute pairwise distances between two sets of points
# These exact patterns appear in attention mechanisms later
```

What to build in `04_linear_algebra.py`:

```python
# Implement:
# - Matrix inverse using np.linalg.inv and verify A @ A_inv = I
# - PCA from scratch (compute eigenvectors manually)
# - Solve a linear system Ax = b two ways: np.linalg.solve and manually
```

**Commit message:** `feat: broadcasting patterns and linear algebra ops from scratch`

---

#### Day 5–7: OOP patterns for ML

What to build in `phase1_foundations/oop_patterns/`:

```python
# 01_classes_and_modules.py
# Build a small Layer base class:
class Layer:
    def __init__(self): ...
    def forward(self, x): raise NotImplementedError
    def __call__(self, x): return self.forward(x)

# Then subclass it:
class LinearLayer(Layer): ...
class ReLU(Layer): ...

# 02_inheritance.py
# Build a Sequential container that chains layers:
class Sequential:
    def __init__(self, *layers): ...
    def forward(self, x): ...

# 03_custom_iterator.py
# Build a DataLoader class that batches and shuffles a NumPy dataset
class SimpleDataLoader:
    def __init__(self, X, y, batch_size, shuffle=True): ...
    def __iter__(self): ...
    def __next__(self): ...
```

**Commit message:** `feat: OOP layer classes and simple sequential container`

---

### Week 2 — Backpropagation from Scratch

This is the Phase 1 milestone. Every line you write here will deepen your understanding of PyTorch autograd in Phase 2.

#### Day 1–3: Forward pass

What to build in `phase1_foundations/backprop_from_scratch/01_forward_pass.py`:

```python
# Implement a 2-layer neural network forward pass:
# Input (any size) -> Linear(64) -> ReLU -> Linear(10) -> Softmax
# Everything as NumPy operations, no torch, no keras
# Use the OOP patterns from week 1
```

#### Day 4–5: Backward pass

What to build in `02_backward_pass.py`:

```python
# Implement gradients for each operation:
# - dL/dW for linear layer (chain rule)
# - dL/db for bias
# - dReLU/dx (the derivative is just a mask)
# - Cross-entropy gradient
# Write unit tests: verify your gradients numerically
# (perturb weights by epsilon, check finite differences match)
```

#### Day 6–7: Full training loop — the milestone

What to build in `04_two_layer_net.py`:

```python
# Full training loop:
# 1. Generate synthetic XOR data
# 2. Forward pass
# 3. Compute cross-entropy loss
# 4. Backward pass (your implementation)
# 5. Update weights
# 6. Plot loss curve
# Network must learn XOR — if it converges, you're done
```

**Commit message:** `feat(milestone): two-layer net with manual backprop learns XOR`

> This commit is your Phase 1 completion marker. The README progress tracker should be updated here.

---

### Environment Setup

Create these files at the root of the repository before writing any code:

**`environment.yml`**
```yaml
name: qmlhep
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - matplotlib
  - jupyter
  - pip
  - pip:
    - torch
    - torchvision
    - pennylane
    - torch_geometric
    - scikit-learn
    - tqdm
```

**`requirements.txt`**
```
numpy>=1.24
matplotlib>=3.7
torch>=2.1
torchvision>=0.16
pennylane>=0.35
torch_geometric>=2.4
scikit-learn>=1.3
tqdm>=4.65
jupyter>=1.0
```

**`.gitignore`**
```
__pycache__/
*.py[cod]
.ipynb_checkpoints/
*.egg-info/
dist/
build/
.env
.venv
env/
venv/
.DS_Store
*.pt
*.pth
data/raw/
*.npz
*.npy
wandb/
```

---

### Git Workflow for This Phase

Follow this commit pattern throughout Phase 1:

| Prefix | When to use |
|--------|------------|
| `feat:` | New implementation or exercise |
| `fix:` | Correcting a bug in your code |
| `docs:` | Updating README or notes |
| `test:` | Adding numerical verification |
| `feat(milestone):` | Completing a phase milestone |

Push at least once every 2 days. Commit history matters — an active repo with small, meaningful commits is better than one large dump.

---

### Progress Tracker

Update this table in your `README.md` as you go:

| Phase | Status | Milestone | Date Completed |
|-------|--------|-----------|----------------|
| Phase 1: Python & NumPy | 🔄 In progress | Backprop from scratch | — |
| Phase 2: PyTorch | ⬜ Not started | CNN >80% CIFAR-10 | — |
| Phase 3: Transformers | ⬜ Not started | Mini-ViT on MNIST | — |
| Phase 4: Quantum basics | ⬜ Not started | Bell state + VQC | — |
| Phase 5: PennyLane | ⬜ Not started | Hybrid Iris classifier | — |
| Phase 6: HEP papers | ⬜ Not started | Reproduce baseline | — |
| Phase 7: QVIT prototype | ⬜ Not started | Working QVIT | — |
| Task I | ⬜ Not started | Both circuits submitted | — |
| Task II | ⬜ Not started | 2 GNN architectures | — |
| Task III | ⬜ Not started | Commentary written | — |
| Task VIII | ⬜ Not started | ViT + QVIT design | — |

---

### Resources for Phase 1

| Resource | Link | Used for |
|----------|------|----------|
| CS50P | cs50.harvard.edu/python | Python foundations |
| 100 NumPy Exercises | github.com/rougier/numpy-100 | NumPy practice |
| Corey Schafer OOP | youtube.com/@coreyms | OOP patterns |
| 3Blue1Brown Neural Nets | youtube.com/3b1b | Backprop intuition |

---

*Last updated: June 2026*
*Next phase: Phase 2 — Deep Learning with PyTorch*
