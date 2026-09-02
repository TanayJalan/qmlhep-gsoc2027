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
print(f".flatten() shape: {x_3d.flatten().shape}")

# squeeze / unsqueeze (PyTorch names for squeeze / expand_dims)
img  = torch.randn(28, 28)
img_batched = img.unsqueeze(0)          # add batch dim at position 0
print(f"\n  Image:          {img.shape}")
print(f"  .unsqueeze(0):  {img_batched.shape}   <- add batch dim")
print(f"  .squeeze(0):    {img_batched.squeeze(0).shape}  <- remove it")

# permute (torch version of np.transpose for N-D)
feat = torch.randn(2, 3, 4)
print(f"\n  (2,3,4).permute(0,2,1): {feat.permute(0,2,1).shape}")

# Section - 4 Math operations

print(' Section 4')

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"  a:{a.tolist()}")
print(f"  b:           {b.tolist()}")
print(f"  a + b:       {(a + b).tolist()}")
print(f"  a * b:       {(a * b).tolist()}   (element-wise)")
print(f"  a @ b:       {(a @ b).item()}     (dot product)")
print(f"  a.dot(b):    {a.dot(b).item()}")

# Matrix multiply
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B                               # (3, 5)
print(f"\n  (3,4) @ (4,5) = {C.shape}")

# Batched matmul
A3 = torch.randn(8, 3, 4)
B3 = torch.randn(8, 4, 5)
C3 = torch.bmm(A3, B3)                 # (8, 3, 5)
print(f"bmm (8,3,4) × (8,4,5) = {C3.shape}")

# Reduction ops
t = torch.randn(4, 5)
print(f"\n  t.sum():              {t.sum().item():.4f}")
print(f"  t.mean():             {t.mean().item():.4f}")
print(f"  t.sum(dim=0).shape:   {t.sum(dim=0).shape}   (per column)")
print(f"  t.mean(dim=1).shape:  {t.mean(dim=1).shape}  (per row)")
print(f"  keepdim=True shape:   {t.mean(dim=1, keepdim=True).shape}")


t2 = torch.ones(3)
t2.add_(1.0)                            # t2 += 1 in-place
print(f"\n  In-place add_ result: {t2.tolist()}")
print(f" Avoid in-place on tensors that require grad (breaks autograd)")

print(' Section -5')

t_cpu = torch.randn(3, 3)
t_dev = t_cpu.to(DEVICE)
print(f"  CPU tensor device:    {t_cpu.device}")
print(f"  After .to(device):    {t_dev.device}")

# Checking if a tensor is on the right device
print(f"\n  t_dev.is_cuda:  {t_dev.is_cuda}")
print(f"  t_dev.device:   {t_dev.device}")

# Back to CPU for NumPy conversion
t_back = t_dev.cpu()
arr    = t_back.numpy()
print(f"\n  .cpu().numpy() -> np.ndarray shape: {arr.shape}")

# section 6

print('Section 6')


arr = np.array([1.0, 2.0, 3.0])
t   = torch.from_numpy(arr)
print(f"  np -> torch: {t.tolist()}  dtype: {t.dtype}")

# Shared memory: modifying arr changes t
arr[0] = 99.0
print(f"After arr[0]=99: tensor = {t.tolist()}  shared memory!")

# Safe conversion back
t2  = torch.randn(4)
arr2 = t2.detach().numpy()
print(f"\n  torch -> np (detach): {arr2.round(4)}")

