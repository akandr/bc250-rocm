#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Figures for the PyTorch, rocBLAS and multi-stream measurements, repo style, grayscale.

Kept separate from make_figures.py, which carries transcribed literals from older campaigns. These are
driven from the JSON that scripts/torch_bench.py writes, so a re-run updates the plots instead of requiring the numbers to be copied by hand, which is how a figure in this repository once ended up
quoting a different run from the paragraph beside it.

    python3 make_gpgpu_figures.py [logs/alu-rates-recheck-2026-09-25/torch_bench_recheck.json]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_d = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_d, "..")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

VK = "#2b2b2b"
HIP = "#9a9a9a"
ACC = "#555555"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 150,
})

# Ceilings measured on this board, logs/alu-rates-recheck-2026-09-25/. The earlier pair,
# 4.74 and 14.40, was taken with the governor below its cap and gave a 3x fp16 advantage that the
# instruction costs do not support; the true ratio is 2.00.
CEIL_F32 = 6.52      # TFLOP/s, logs/alu-rates-recheck-2026-09-25 (clock verified)
CEIL_F16 = 13.02     # TFLOP/s, packed, exactly 2x fp32
CEIL_BW = 432.0      # GB/s

# The hand-written packed-fp16 kernel, logs/pkf16-vs-rocblas-2026-09-25/pk_run3.txt, TFLOP/s.
PKF16 = {512: 1.64, 1024: 5.14, 2048: 4.98, 4096: 7.66, 8192: 8.91}

# rocBLAS, logs/alu-rates-recheck-2026-09-25/rocblas_recheck.out, GFLOP/s.
# Transcribed from that file and checked against it; an earlier set of literals here came from
# somewhere else and differed by up to 2 percent, which is what scripts/audit_figures.py exists for.
ROCBLAS = {
    "SGEMM fp32": {512: 1440.9, 1024: 2530.9, 2048: 2989.7, 4096: 4520.6, 8192: 4611.9},
    "HGEMM fp16": {512: 1714.2, 1024: 3375.4, 2048: 2866.0, 4096: 4465.7, 8192: 4637.1},
}

# The multi-stream option across the model set, logs/campaign-graphopt-2026-09-24/
GRAPH_OPT = [
    ("qwen2.5-1.5B", 196.7, 213.1, 212.3),
    ("qwen3-8B", 38.5, 39.5, 39.0),
    ("deepseek-14B", 32.6, 33.6, 35.1),
    ("qwen3-14B", 32.7, 33.7, 34.7),
    ("35B-A3B MoE", 71.2, 70.8, 86.9),
    ("qwen3.8-27B", 15.2, 15.1, 17.6),
]


def load(path):
    with open(path) as fh:
        return json.load(fh)


def fig_gemm_ceiling(rows):
    """Achieved GEMM rate against the two ceilings: fp32 reaches its roof, fp16 does not."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    styles = {"float32": ("-o", VK), "float16": ("-D", HIP), "bfloat16": ("--s", "#c0c0c0")}
    for dt, (mk, col) in styles.items():
        pts = sorted([(r["n"], r["tflops"]) for r in rows if r["kind"] == "gemm" and r["dtype"] == dt])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, color=col, ms=5, label=f"PyTorch {dt}")
    rb = sorted(ROCBLAS["SGEMM fp32"].items())
    ax.plot([k for k, _ in rb], [v / 1000 for _, v in rb], ":", color=VK, lw=1.2, label="rocBLAS SGEMM")
    rb = sorted(ROCBLAS["HGEMM fp16"].items())
    ax.plot([k for k, _ in rb], [v / 1000 for _, v in rb], ":", color=HIP, lw=1.2, label="rocBLAS HGEMM")

    pk = sorted(PKF16.items())
    ax.plot([k for k, _ in pk], [v for _, v in pk], "-^", color="#111111", ms=6, lw=1.6,
            label="hand-written packed-fp16 kernel")

    ax.axhline(CEIL_F32, color=VK, lw=1.0, ls="--")
    ax.axhline(CEIL_F16, color=HIP, lw=1.0, ls="--")
    # Ceiling labels ride just under their lines on the right, clear of the legend on the left.
    ax.text(8100, CEIL_F32 - 0.75, f"fp32 ceiling {CEIL_F32}", fontsize=7.5, color=VK, ha="right")
    ax.text(8100, CEIL_F16 - 0.85, f"packed fp16 ceiling {CEIL_F16}", fontsize=7.5,
            color="#6a6a6a", ha="right")
    # The arrow is drawn between the two points it compares; its label sits beside it, not on it.
    ax.annotate("", xy=(8192, 8.91), xytext=(8192, 4.64),
                arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.1))
    ax.text(7400, 6.6, "1.9x", fontsize=8, color="#222222", ha="right", fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["512", "1024", "2048", "4096", "8192"])
    ax.set_xlabel("square GEMM size N"); ax.set_ylabel("TFLOP/s")
    ax.set_ylim(0, 16)
    ax.set_title("fp32 gets close to its ceiling; fp16 does not, and a plain kernel beats the library",
                 fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 0.80))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-gemm-vs-ceiling.png"), bbox_inches="tight")
    print("fig-gemm-vs-ceiling.png")


def fig_roofline(rows):
    """Every measured kernel against the memory roof and the two compute roofs."""
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ai = np.logspace(-2, 4, 400)
    ax.plot(ai, np.minimum(CEIL_BW * ai / 1000, CEIL_F16), color=HIP, lw=1.3, ls="--",
            label=f"roof, packed fp16 ({CEIL_F16} TFLOP/s)")
    ax.plot(ai, np.minimum(CEIL_BW * ai / 1000, CEIL_F32), color=VK, lw=1.5,
            label=f"roof, fp32 ({CEIL_F32} TFLOP/s) over memory at {CEIL_BW:.0f} GB/s")

    get = lambda k, **kw: next((r for r in rows if r["kind"] == k
                                and all(r.get(a) == v for a, v in kw.items())), None)

    # GEMM: 2N^3 flops over 3N^2 elements, so intensity is N/(3*bytes)
    pts = []
    for dt, mk, col, bs in (("float32", "o", VK, 4), ("float16", "D", HIP, 2),
                            ("bfloat16", "s", "#c6c6c6", 2)):
        r = get("gemm", dtype=dt, n=8192)
        if r:
            pts.append((8192 / (3.0 * bs), r["tflops"], dt, mk, col))
    for x, y, dt, mk, col in pts:
        ax.plot([x], [y], mk, color=col, ms=7, zorder=5)
    # one label per cluster, placed clear of the markers
    if pts:
        ax.annotate("GEMM N=8192\nfp32 / fp16 / bf16", (pts[0][0] * 0.92, 2.1),
                    fontsize=7.5, color="#333333", ha="center")

    r = get("bw", n=1 << 26)
    if r:
        x, y = 1.0 / 12.0, r["gbs"] / 12.0 / 1000
        ax.plot([x], [y], "s", color=VK, ms=7, zorder=5)
        ax.annotate(f"elementwise add\n{r['gbs']:.0f} GB/s, {r['pct_peak']:.0f}% of the roof",
                    (x * 1.35, y * 0.72), fontsize=7.5, color="#333333")

    r = get("attn", dtype="float32", seq=2048)
    if r:
        ax.plot([60.0], [r["tflops"]], "^", color=VK, ms=7, zorder=5)
        ax.annotate("attention fp32, seq 2048", (72.0, r["tflops"] * 0.80),
                    fontsize=7.5, color="#333333")

    # the headline: fp16 GEMM sits far under a roof three times higher
    rf = get("gemm", dtype="float16", n=8192)
    if rf:
        xf = 8192 / 6.0
        ax.annotate("", xy=(xf, CEIL_F16), xytext=(xf, rf["tflops"]),
                    arrowprops=dict(arrowstyle="<->", color="#777777", lw=1.1))
        # Sits in the empty band between the two roofs and to the left of the arrow; to the right
        # of it the fp32 roof runs straight through the second line of the label.
        ax.annotate("fp16 GEMM runs at about\na third of its own roof", (xf * 0.45, 8.3),
                    fontsize=7.5, color="#444444", ha="right")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP per byte)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_xlim(0.03, 8000); ax.set_ylim(0.02, 30)
    ax.set_title("Roofline, with both roofs measured on this board", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-roofline.png"), bbox_inches="tight")
    print("fig-roofline.png")


def fig_training(rows):
    """fp32 scales with batch; fp16 autocast goes the wrong way."""
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for dt, mk, col in (("float32", "-o", VK), ("float16", "-D", HIP)):
        pts = sorted([(r["batch"], r["tok_per_s"] / 1000) for r in rows
                      if r["kind"] == "train" and r["dtype"] == dt])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], mk, color=col, ms=6,
                    label="fp32" if dt == "float32" else "fp16 autocast")
            # fp32 labels sit above their points and fp16 below, because at batch 1 the two
            # series are close enough that same-side labels overlap.
            dy = 7 if dt == "float32" else -13
            for x, y in pts:
                ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, dy),
                            fontsize=7, ha="center", color="#333333")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16]); ax.set_xticklabels(["1", "4", "16"])
    ax.set_xlabel("batch size"); ax.set_ylabel("training throughput (ktok/s)")
    ax.set_title("Training a small transformer: fp16 autocast is a regression here", fontsize=9)
    ax.set_ylim(0, 40)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-training-precision.png"), bbox_inches="tight")
    print("fig-training-precision.png")


def fig_graph_opt():
    """What the multi-stream option is worth, and where it does nothing."""
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    names = [g[0] for g in GRAPH_OPT][::-1]
    off = np.array([g[1] for g in GRAPH_OPT][::-1])
    on = np.array([g[2] for g in GRAPH_OPT][::-1])
    vk = np.array([g[3] for g in GRAPH_OPT][::-1])
    y = np.arange(len(names))
    ax.barh(y + 0.20, off / vk, height=0.36, color="#cfcfcf", label="ROCm as shipped / Vulkan")
    ax.barh(y - 0.20, on / vk, height=0.36, color=HIP, label="ROCm + GRAPH_OPT / Vulkan")
    ax.axvline(1.0, color=VK, lw=1.2)
    for i, (a, b) in enumerate(zip(off / vk, on / vk)):
        ax.text(b + 0.012, i - 0.20, f"{b:.2f}", va="center", fontsize=7, color="#222222")
        ax.text(a + 0.012, i + 0.20, f"{a:.2f}", va="center", fontsize=7, color="#666666")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0.7, 1.13); ax.set_xlabel("decode rate relative to Vulkan (1.0 = parity)")
    ax.set_title("GGML_CUDA_GRAPH_OPT=1: gains where it forks streams, nothing where it cannot",
                 fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-graph-opt.png"), bbox_inches="tight")
    print("fig-graph-opt.png")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "logs", "alu-rates-recheck-2026-09-25", "torch_bench_recheck.json")
    rows = load(path)
    fig_gemm_ceiling(rows)
    fig_roofline(rows)
    fig_training(rows)
    fig_graph_opt()
    return 0


if __name__ == "__main__":
    sys.exit(main())
