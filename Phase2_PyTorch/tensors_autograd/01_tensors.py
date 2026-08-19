"""
PyTorch tensors are NumPy arrays with two superpowers:
    1. They can live on a GPU (or Apple MPS) for fast computation
    2. They track operations for automatic differentiation
"""

import numpy as np
import torch

print(f"PyTorch version: {torch.__version__}")

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    Device = torch.device('cuda')
else:
    Device = torch.device('cpu')

print(f'Device: {DEVICE}\n')

print('Section1 - tensor create')
np_zeros = np.zeros((3, 4))
pt_zeros = torch.zeros(3, 4)

print(f'np.zeros(3,4) -> torch.zeros(3,4)')
print(f' shape: {pt_zeros.shape} dtype: {pt_zeros.dtype}')

np_rand = np.random.randn(3, 3)
pt_rand = torch.randn(3, 3)

print(f"\n np.random.randn -> torch.randn(3, 3)")
print(f"{pt_rand.round(decimals=3)}")

data = [[1.0, 2.0], [3.0, 4.0]]
pt_from_list = torch.tensor(data)
pt_from_np = torch.from_numpy(np.array(data))   # zero-copy when possible
print(f"\ntorch.tensor(list) -> {pt_from_list.tolist()}")
print(f"torch.from_numpy() -> {pt_from_np.tolist()}")

pt_arange  = torch.arange(0, 10, 2)
pt_linspace = torch.linspace(0, 1, 5)
print(f"\n arange(0,10,2): {pt_arange.tolist()}")
print(f"linspace(0,1,5): {pt_linspace.tolist()}")

# Eye
pt_eye = torch.eye(4)
print(f"\n torch.eye(4):\n{pt_eye}")

# Metadata
t = torch.randn(4, 3)
print(f"\n Tensor metadata:")
print(f"shape:{t.shape}")
print(f"dtype:{t.dtype}")
print(f"device:{t.device}")
print(f"ndim:{t.ndim}")
print(f"numel(): {t.numel()}")


# section 2 Indexing and slicing
print('Section 2- Indexing & slicing')

B = torch.tensor([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12]
], dtype=torch.float32)

print(f' B:\n{B}')
print(f'\n B[0,:] = {B[0,:].tolist()}')
print(f'\n B[:, 1] = {B[:,1].tolist()}')
print(f'\n B[1:, 2:] = {B[1:,2:].tolist()}')

mask = B>6
print(f'\n B[B>6] = {B[mask].tolist()}')

scalar_tensor = B[0,0]
print(f'\n B[0,0] = {scalar_tensor}')
print(f' B[0,0].item() = {scalar_tensor.item()}')


print('Section3- reshaping')

x = torch.arange(24, dtype=torch.float32)
print(f' arange(24): {x.tolist()}')

x_2d  = x.view(4, 6)
x_3d  = x.reshape(2, 3, 4)
print(f"\n .view(4, 6)       shape: {x_2d.shape}")
print(f"  .reshape(2, 3, 4) shape: {x_3d.shape}")

# -1 inference works the same as NumPy
x_auto = x.reshape(6, -1)
print(f"  .reshape(6, -1)   shape: {x_auto.shape}  (-1 inferred as 4)")

# flatten
print(f"  .flatten()        shape: {x_3d.flatten().shape}")

# squeeze / unsqueeze (PyTorch names for squeeze / expand_dims)
img  = torch.randn(28, 28)
img_batched = img.unsqueeze(0)          # add batch dim at position 0
print(f"\n  Image:          {img.shape}")
print(f"  .unsqueeze(0):  {img_batched.shape}   ← add batch dim")
print(f"  .squeeze(0):    {img_batched.squeeze(0).shape}  ← remove it")

# permute (torch version of np.transpose for N-D)
feat = torch.randn(2, 3, 4)
print(f"\n  (2,3,4).permute(0,2,1): {feat.permute(0,2,1).shape}")

