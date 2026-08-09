#!/usr/bin/env python3
"""PyTorch matmul benchmark for the BC-250 (native gfx1013 rocBLAS grafted
into the official torch wheel).

Default mode preallocates the output and host buffer, which is the discipline
that makes sustained GPU matmul work on this board. --churn switches to the
naive pattern (allocate and free the output every iteration), which faults
after 20 to 40 iterations here; run it to reproduce that failure mode.

The official wheel ships no gfx1013 elementwise kernels, so tensors are
created on the CPU and moved; only the matmul itself runs on the GPU.

Usage: torch_matmul_bench.py [N] [iters] [--churn]
"""
import sys, time
import torch

args = [a for a in sys.argv[1:] if not a.startswith("--")]
N = int(args[0]) if args else 4096
iters = int(args[1]) if len(args) > 1 else 30
churn = "--churn" in sys.argv

assert torch.cuda.is_available(), "no GPU"
dev = torch.device("cuda")
print(f"device: {torch.cuda.get_device_name(0)}  N={N} iters={iters} "
      f"mode={'churn' if churn else 'preallocated'}", flush=True)

ac, bc = torch.randn(N, N), torch.randn(N, N)
a, b = ac.to(dev), bc.to(dev)
ref = ac[:8, :] @ bc[:, :8]
c = None if churn else torch.empty(N, N, device=dev)
host = torch.empty(N, N)

bad = 0
t0 = time.time()
for it in range(iters):
    if churn:
        c = a @ b                       # new allocation every iteration
    else:
        torch.mm(a, b, out=c)           # reused output buffer
    torch.cuda.synchronize()
    host.copy_(c)
    err = (host[:8, :8] - ref[:, :8]).abs().max().item() / (ref.abs().max().item() + 1e-9)
    if err > 1e-3:
        bad += 1
        print(f"iter {it} WRONG rel_err={err:.2e}", flush=True)
dt = time.time() - t0
print(f"DONE iters={iters} bad={bad} {dt:.1f}s {2*N**3*iters/dt/1e9:.0f} GFLOP/s", flush=True)
sys.exit(0 if bad == 0 else 3)
