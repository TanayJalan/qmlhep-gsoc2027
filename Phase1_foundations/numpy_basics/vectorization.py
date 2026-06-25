import numpy as np
import time

def benchmark(fn, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times), result

def print_header(title):
    print("\n" + "="*55)
    print(f" {title}")
    print("="*55)

def print_timing(label, t, baseline=None):
    msg = f"  {label:<30} {t*1000:>8.3f} ms"
    if baseline is not None:
        speedup = baseline / t
        msg += f" ({speedup:.1f}x faster)"
    print(msg)
    
print_header("Operation 1 — Elementwise Multiply")
print("task: multiply two arrays of 1,000,000 floats elementwise")
print("ml use: scaling activations, applying masks, weight updates")

N = 1_000_000
a = np.random.randn(N)
b = np.random.randn(N)

def loop_multiply():
    result = [0.0] * N
    for i in range(N):
        result[i] =a[i] * b[i]
    return result

def numpy_multiply():
    return a * b

def broadcast_multiply():
    return np.multiply(a,b)


t_loop, _ = benchmark(loop_multiply,repeats=3)
t_numpy, r_np = benchmark(numpy_multiply)
t_bcast, r_bc = benchmark(broadcast_multiply)

