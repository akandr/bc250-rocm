#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""PyTorch throughput on gfx1013, measured against the board's own ceilings.

logs/torch-train-2026-08-19/ and logs/fedora44-validation-2026-09-15/ establish that PyTorch is
*correct* on this board. Neither says how fast it is, and PyTorch is the main reason to run ROCm here
at all, since Vulkan has no equivalent. This measures it.

Absolute throughput figures mean little on a part nobody has a reference for, so every number is also
reported as a fraction of a ceiling this repository measured directly on this silicon
(logs/alu-rates-2026-09-19/, logs/fedora44-benchmarks-2026-09-15/gpgpu/):

    v_fma_f32        6.52 TFLOP/s     the fp32 arithmetic ceiling
    v_pk_fma_f16    13.02 TFLOP/s     the packed-fp16 ceiling, exactly twice fp32
    streaming read 432    GB/s        the memory ceiling

Method: buffers allocated once and reused so the caching allocator is not in the measurement; at least
WARMUP untimed iterations; REPS timed iterations each bracketed by torch.cuda.synchronize(); the median
reported with the full range, because a single reading on this board is not trustworthy. Every GEMM is
checked against a CPU reference once, at the smallest size, before timing.

Run under the source-built PyTorch, not Fedora's package, which has no gfx1013 kernels and dumps core:
    ~/torchbench-venv/bin/python torch_bench.py
"""

import argparse
import json
import statistics
import subprocess
import sys
import time

import torch

# Ceilings measured on this board, not vendor figures.
CEIL_FP32_TFLOPS = 6.52   # logs/alu-rates-recheck-2026-09-25, clock verified at 1500 MHz
CEIL_FP16_TFLOPS = 13.02  # same run; packed fp16 is 2.00x fp32, not 3x
CEIL_BW_GBS = 432.0

WARMUP = 3
REPS = 7


def edge_temp():
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "edge" in line:
                return line.split()[1]
    except Exception:
        pass
    return "?"


def timed(fn, warmup=WARMUP, reps=REPS):
    """Median and range of seconds per call, synchronised."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def gemm(dev, sizes, dtypes):
    print("\n## GEMM, square N x N, 2*N^3 flops")
    print(f"{'dtype':9s} {'N':>6s} {'ms':>9s} {'TFLOP/s':>9s} {'ceiling':>9s} {'% peak':>7s} {'spread':>7s}")
    rows = []
    for dtype in dtypes:
        ceil = CEIL_FP32_TFLOPS if dtype == torch.float32 else CEIL_FP16_TFLOPS
        for n in sizes:
            try:
                a = torch.randn(n, n, device=dev, dtype=dtype)
                b = torch.randn(n, n, device=dev, dtype=dtype)
                out = torch.empty(n, n, device=dev, dtype=dtype)
            except RuntimeError as e:
                print(f"{_dt(dtype):9s} {n:6d}   skipped: {str(e)[:40]}")
                continue
            med, lo, hi = timed(lambda: torch.matmul(a, b, out=out))
            tflops = 2.0 * n ** 3 / med / 1e12
            print(f"{_dt(dtype):9s} {n:6d} {med*1e3:9.2f} {tflops:9.3f} {ceil:9.2f} "
                  f"{tflops/ceil*100:6.1f}% {(hi-lo)/med*100:6.1f}%")
            rows.append(dict(kind="gemm", dtype=_dt(dtype), n=n, ms=med * 1e3,
                             tflops=tflops, ceiling=ceil, pct_peak=tflops / ceil * 100))
            del a, b, out
            torch.cuda.empty_cache()
    return rows


def bandwidth(dev, sizes):
    print("\n## Memory bound: c = a + b, 3 arrays touched (2 read, 1 write), fp32")
    print(f"{'elements':>10s} {'MiB moved':>10s} {'ms':>9s} {'GB/s':>8s} {'% of 432':>9s} {'spread':>7s}")
    rows = []
    for n in sizes:
        a = torch.randn(n, device=dev)
        b = torch.randn(n, device=dev)
        c = torch.empty(n, device=dev)
        med, lo, hi = timed(lambda: torch.add(a, b, out=c))
        moved = 3 * n * 4
        gbs = moved / med / 1e9
        print(f"{n:10d} {moved/2**20:10.1f} {med*1e3:9.3f} {gbs:8.1f} {gbs/CEIL_BW_GBS*100:8.1f}% "
              f"{(hi-lo)/med*100:6.1f}%")
        rows.append(dict(kind="bw", n=n, ms=med * 1e3, gbs=gbs, pct_peak=gbs / CEIL_BW_GBS * 100))
        del a, b, c
        torch.cuda.empty_cache()
    return rows


class Block(torch.nn.Module):
    """One pre-norm transformer block, the shape that actually runs in practice."""

    def __init__(self, d, heads):
        super().__init__()
        self.n1 = torch.nn.LayerNorm(d)
        self.att = torch.nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = torch.nn.LayerNorm(d)
        self.ff = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.att(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))


def training(dev, d_model, heads, layers, seq, batches, dtypes):
    print(f"\n## Training step, {layers} pre-norm transformer blocks, d_model={d_model}, "
          f"heads={heads}, seq={seq}, Adam")
    print(f"{'precision':10s} {'batch':>6s} {'ms/step':>9s} {'steps/s':>8s} {'ktok/s':>8s} {'spread':>7s}")
    rows = []
    for dtype in dtypes:
        for bs in batches:
            torch.manual_seed(0)
            model = torch.nn.Sequential(*[Block(d_model, heads) for _ in range(layers)]).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            x = torch.randn(bs, seq, d_model, device=dev)
            amp = dtype is not torch.float32

            def step():
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype, enabled=amp):
                    loss = model(x).square().mean()
                loss.backward()
                opt.step()

            try:
                med, lo, hi = timed(step, warmup=3, reps=5)
            except RuntimeError as e:
                print(f"{_dt(dtype):10s} {bs:6d}   skipped: {str(e)[:40]}")
                del model, opt, x
                torch.cuda.empty_cache()
                continue
            toks = bs * seq
            print(f"{_dt(dtype):10s} {bs:6d} {med*1e3:9.1f} {1/med:8.2f} {toks/med/1e3:8.1f} "
                  f"{(hi-lo)/med*100:6.1f}%")
            rows.append(dict(kind="train", dtype=_dt(dtype), batch=bs, ms=med * 1e3,
                             steps_per_s=1 / med, tok_per_s=toks / med))
            del model, opt, x
            torch.cuda.empty_cache()
    return rows


def conv2d(dev, dtypes):
    """Conv is the other half of what people run in PyTorch and is untested on this board.

    The rate below counts flops by the direct-convolution formula, 2*out*C*9. A library that uses
    Winograd or an FFT does algebraically fewer multiplies than that, so the figure can exceed the
    arithmetic ceiling; it measures useful work delivered, not instructions issued, and a value above
    100 percent is evidence the library is not doing direct convolution, not a broken timer.
    """
    print("\n## conv2d, 3x3, stride 1, pad 1, NCHW")
    print("   rate is effective flops by the direct-convolution count; >100% means an algorithmically")
    print("   cheaper method (Winograd or FFT), not a faster machine")
    print(f"{'dtype':9s} {'N':>3s} {'C':>4s} {'H':>4s} {'ms':>9s} {'eff TF/s':>9s} {'ceiling':>9s} {'vs ceil':>7s} {'spread':>7s}")
    rows = []
    shapes = [(8, 64, 128), (8, 256, 64), (16, 128, 64)]
    for dtype in dtypes:
        ceil = CEIL_FP32_TFLOPS if dtype == torch.float32 else CEIL_FP16_TFLOPS
        for (bs, ch, hw) in shapes:
            try:
                x = torch.randn(bs, ch, hw, hw, device=dev, dtype=dtype)
                w = torch.randn(ch, ch, 3, 3, device=dev, dtype=dtype)
            except RuntimeError as e:
                print(f"{_dt(dtype):9s} {bs:3d} {ch:4d} {hw:4d}   skipped: {str(e)[:32]}")
                continue
            try:
                med, lo, hi = timed(lambda: torch.nn.functional.conv2d(x, w, padding=1))
            except RuntimeError as e:
                print(f"{_dt(dtype):9s} {bs:3d} {ch:4d} {hw:4d}   failed: {str(e)[:32]}")
                del x, w
                torch.cuda.empty_cache()
                continue
            # 2 flops per multiply-add, over output elements x input channels x kernel area
            flops = 2.0 * bs * ch * hw * hw * ch * 9
            tf = flops / med / 1e12
            print(f"{_dt(dtype):9s} {bs:3d} {ch:4d} {hw:4d} {med*1e3:9.2f} {tf:9.3f} {ceil:9.2f} "
                  f"{tf/ceil*100:6.1f}% {(hi-lo)/med*100:6.1f}%")
            rows.append(dict(kind="conv", dtype=_dt(dtype), bs=bs, ch=ch, hw=hw, ms=med * 1e3,
                             tflops=tf, pct_peak=tf / ceil * 100))
            del x, w
            torch.cuda.empty_cache()
    return rows


def attention(dev, dtypes):
    """scaled_dot_product_attention: this build reports no memory-efficient kernel, so it falls back."""
    print("\n## scaled_dot_product_attention, batch 8, 8 heads, head_dim 64")
    print(f"{'dtype':9s} {'seq':>6s} {'ms':>9s} {'TFLOP/s':>9s} {'ceiling':>9s} {'% peak':>7s} {'spread':>7s}")
    rows = []
    for dtype in dtypes:
        ceil = CEIL_FP32_TFLOPS if dtype == torch.float32 else CEIL_FP16_TFLOPS
        for seq in (512, 1024, 2048):
            q = torch.randn(8, 8, seq, 64, device=dev, dtype=dtype)
            k = torch.randn(8, 8, seq, 64, device=dev, dtype=dtype)
            v = torch.randn(8, 8, seq, 64, device=dev, dtype=dtype)
            try:
                med, lo, hi = timed(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
            except RuntimeError as e:
                print(f"{_dt(dtype):9s} {seq:6d}   failed: {str(e)[:36]}")
                del q, k, v
                torch.cuda.empty_cache()
                continue
            # two matmuls: QK^T and (softmax)V
            flops = 2.0 * 2.0 * 8 * 8 * seq * seq * 64
            tf = flops / med / 1e12
            print(f"{_dt(dtype):9s} {seq:6d} {med*1e3:9.2f} {tf:9.3f} {ceil:9.2f} "
                  f"{tf/ceil*100:6.1f}% {(hi-lo)/med*100:6.1f}%")
            rows.append(dict(kind="attn", dtype=_dt(dtype), seq=seq, ms=med * 1e3,
                             tflops=tf, pct_peak=tf / ceil * 100))
            del q, k, v
            torch.cuda.empty_cache()
    return rows


def cpu_reference(n):
    """The same GEMM on the CPU, twice, because the two answers differ by 44x.

    This PyTorch is a source build with neither MKL nor MKLDNN, so torch's own CPU GEMM falls back to a
    generic kernel and is not what this CPU can do. numpy on the same machine links scipy-openblas and
    is the fair reference; torch-CPU is reported beside it only to show how far a build without a tuned
    BLAS falls, since quoting the GPU against it would flatter the GPU by a factor of forty.
    """
    out = {}

    a = torch.randn(n, n)
    b = torch.randn(n, n)
    for _ in range(2):
        a @ b
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        a @ b
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    out["torch"] = (2.0 * n ** 3 / med / 1e12, med)

    try:
        import numpy as np
        an = np.random.rand(n, n).astype(np.float32)
        bn = np.random.rand(n, n).astype(np.float32)
        for _ in range(2):
            an @ bn
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            an @ bn
            ts.append(time.perf_counter() - t0)
        med = statistics.median(ts)
        out["numpy"] = (2.0 * n ** 3 / med / 1e12, med)
    except Exception:
        pass
    return out


def _dt(d):
    return str(d).replace("torch.", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the rows here")
    ap.add_argument("--quick", action="store_true", help="smaller sweep")
    args = ap.parse_args()

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}  hip {torch.version.hip}")
    print(f"device {torch.cuda.get_device_name(0)}  arch {props.gcnArchName}  "
          f"CUs {props.multi_processor_count}  vram {props.total_memory/2**30:.1f} GiB")
    print(f"ceilings measured on this board: fp32 {CEIL_FP32_TFLOPS} TFLOP/s, "
          f"packed fp16 {CEIL_FP16_TFLOPS} TFLOP/s, read {CEIL_BW_GBS} GB/s")
    print(f"edge {edge_temp()} at start; warmup {WARMUP}, {REPS} timed reps, median of each")

    # correctness before speed, at the smallest size
    a = torch.randn(512, 512, device=dev)
    b = torch.randn(512, 512, device=dev)
    err = ((a @ b).cpu() - (a.cpu() @ b.cpu())).abs().max().item()
    print(f"fp32 GEMM against CPU at N=512: max abs error {err:.2e}")
    del a, b

    sizes = [512, 1024, 2048] if args.quick else [512, 1024, 2048, 4096, 8192]
    bw_sizes = [1 << 20, 1 << 24] if args.quick else [1 << 20, 1 << 22, 1 << 24, 1 << 26]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    rows = []
    rows += gemm(dev, sizes, dtypes)
    rows += bandwidth(dev, bw_sizes)
    rows += conv2d(dev, [torch.float32, torch.float16])
    rows += attention(dev, [torch.float32, torch.float16])
    rows += training(dev, 512, 8, 4, 256,
                     [1, 4] if args.quick else [1, 4, 16],
                     [torch.float32, torch.float16])

    n = 2048
    cpu = cpu_reference(n)
    gpu = [r for r in rows if r["kind"] == "gemm" and r["dtype"] == "float32" and r["n"] == n]
    print(f"\n## CPU reference, the same fp32 GEMM at N={n}, {torch.get_num_threads()} threads")
    for who, (tf, sec) in cpu.items():
        note = "" if who == "numpy" else "   <- no MKL/MKLDNN in this build, not what the CPU can do"
        print(f"{who:6s} {tf:7.3f} TFLOP/s ({sec*1e3:8.1f} ms){note}")
    if gpu and "numpy" in cpu:
        print(f"GPU    {gpu[0]['tflops']:7.3f} TFLOP/s ({gpu[0]['ms']:8.1f} ms)  "
              f"-> {gpu[0]['tflops']/cpu['numpy'][0]:.1f}x the CPU at its best")
        rows.append(dict(kind="cpu", n=n, numpy_tflops=cpu["numpy"][0],
                         torch_cpu_tflops=cpu["torch"][0], gpu_tflops=gpu[0]["tflops"],
                         speedup=gpu[0]["tflops"] / cpu["numpy"][0]))

    print(f"\nedge {edge_temp()} at end")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
