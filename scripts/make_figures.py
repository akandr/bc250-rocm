#!/usr/bin/env python3
"""Figures for the BC-250 ROCm README, grayscale, repo style.

Data below is transcribed from bench-2026-08 logs (llama-bench tables,
sgemm_iter medians). Regenerate: python3 make_figures.py -> ../repo/figures/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

_d = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_d) == "scripts":            # repo/scripts/ -> repo/figures
    OUT = os.path.join(_d, "..", "figures")
else:                                            # validation dir -> ../repo/figures
    OUT = os.path.join(_d, "..", "repo", "figures")
os.makedirs(OUT, exist_ok=True)

VK = "#2b2b2b"   # Vulkan: near-black
HIP = "#9a9a9a"  # ROCm/HIP: mid gray
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 150,
})

# ---------------- data (fill/verify from logs) ----------------
# model label -> (hip_tg, vk_tg, hip_pp512, vk_pp512); None = not measured/failed
# ROCm: fixed stack (integrated flag, KQV precision, RDNA1 macro), fa on,
# native gfx1013 rocBLAS, f32 cuBLAS compute type. Vulkan: same build, same
# boot, fa on. All rows perplexity-gated against Vulkan (2026-08-12 campaign).
MODELS = {
    "qwen2.5 1.5B Q4_K_M":        (113.50, 210.95, 805.61, 1842.18),
    "qwen3 8B Q8_0":              (39.20, 39.14, 241.01, 401.10),
    "deepseek-r1 14B Q4_K_M":     (20.26, 34.52, 95.41, 199.04),
    "qwen3 14B Q4_K_M":           (21.47, 34.15, 97.40, 202.82),
    "qwen3.6 35B-A3B MoE IQ2_M":  (34.34, 86.45, 287.58, 455.40),
    # qwen3.8 decode is tg128, not tg64 like the rows above; decode falls
    # with depth, so the two are not interchangeable. Noted on the axis.
    "qwen3.8 27B UD-IQ3_XXS":     (7.84, 17.18, 69.23, 97.94),
}

# depth ladder (qwen2.5-1.5b): depth -> (hip_tg64, vk_tg64)
DEPTH = {0: (117.59, 210.95), 4096: (103.36, 178.45), 8192: (96.14, 163.76), 16384: (84.38, 143.13), 24576: (74.15, 126.52), 30720: (68.22, 116.50)}

# sgemm: N -> median ms/iter (20 iters, all correct)
SGEMM = {512: 0.2, 1024: 0.8, 2048: 5.7, 4096: 30.0, 8192: 236.0}

# rocBLAS fp16 GEMM throughput by shape (GFLOP/s), f32 accumulate, measured
# 2026-08-17. The layer shapes are what a transformer actually issues; the
# squares are the reference. This is the prefill gap: ROCm prefill lands on the
# layer-shape numbers, Vulkan runs above the whole table because it multiplies
# against quantized weights without materialising an fp16 copy.
GEMM_SHAPES = [
    ("2048^3\nsquare", 4181, "square"),
    ("4096^3\nsquare", 4248, "square"),
    ("1536x512x1536\nattn proj", 2637, "layer"),
    ("8960x512x1536\nffn up", 2838, "layer"),
    ("1536x512x8960\nffn down", 3917, "layer"),
]
# Implied prefill rates are kept out of this figure on purpose: the ROCm
# prefill path for the measured model uses llama.cpp's own quantized kernels
# and issues no rocBLAS calls, so it is not bounded by these bars.

def gflops(n, ms): return 2 * n**3 / (ms / 1e3) / 1e9

# ---------------- fig 1: ROCm vs Vulkan bars ----------------
def fig_backends():
    rows = [(k, v) for k, v in MODELS.items() if v[0] is not None]
    labels = [r[0] for r in rows]
    hip_tg = [r[1][0] for r in rows]; vk_tg = [r[1][1] for r in rows]
    hip_pp = [r[1][2] for r in rows]; vk_pp = [r[1][3] for r in rows]
    y = np.arange(len(rows)); h = 0.36
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 0.7 + 0.62 * len(rows) + 0.8))
    for ax, hipv, vkv, title, log in (
        (a1, hip_tg, vk_tg, "decode (tokens/s; tg64, qwen3.8 tg128)", False),
        (a2, hip_pp, vk_pp, "prefill (pp512, tokens/s, log scale)", True),
    ):
        vkp = [v or 0 for v in vkv]; hipp = [v or 0 for v in hipv]
        ax.barh(y + h/2, vkp, h, color=VK, label="Vulkan (RADV)")
        ax.barh(y - h/2, hipp, h, color=HIP, label="ROCm/HIP (native gfx1013)")
        for yy, v in list(zip(y + h/2, vkv)) + list(zip(y - h/2, hipv)):
            if v: ax.text(v * (1.04 if not log else 1.12), yy, f"{v:g}",
                          va="center", fontsize=7.5, color="#222")
        ax.set_yticks(y); ax.set_yticklabels(labels if ax is a1 else [""] * len(rows))
        ax.invert_yaxis(); ax.set_title(title, fontsize=9)
        if log: ax.set_xscale("log"); ax.set_xlim(1, max(vk_pp) * 4)
        else: ax.set_xlim(0, max(vk_tg) * 1.22)
    a1.legend(loc="center right", fontsize=7.5, frameon=False)
    fig.text(0.01, -0.02, "ROCm: llama.cpp master with the three gfx1013 fixes, flash attention on, "
             "native gfx1013 rocBLAS, f32 cuBLAS compute type. Vulkan: same build and boot. "
             "Every row passes a wikitext perplexity gate against Vulkan.", fontsize=7, color="#444")
    fig.suptitle("llama.cpp on the BC-250 at 40 CU: ROCm/HIP vs Vulkan (same build, same boot config)",
                 fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-rocm-vs-vulkan.png"), bbox_inches="tight")
    print("fig-rocm-vs-vulkan.png")

# ---------------- fig 2: decode vs context depth ----------------
def fig_depth():
    hd = sorted(d for d, v in DEPTH.items() if v[0] is not None)
    vd = sorted(d for d, v in DEPTH.items() if v[1] is not None)
    hip = [DEPTH[d][0] for d in hd]; vk = [DEPTH[d][1] for d in vd]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(vd, vk, "-o", color=VK, label="Vulkan (RADV)", ms=5)
    ax.plot(hd, hip, "-s", color=HIP, label="ROCm/HIP", ms=5)
    for d, v in zip(vd, vk): ax.text(d, v * 1.04, f"{v:g}", fontsize=7.5, ha="center")
    for d, v in zip(hd, hip): ax.text(d, v * 1.06, f"{v:g}", fontsize=7.5, ha="center")
    ax.set_xlabel("context depth before generation (tokens)")
    ax.set_ylabel("decode tokens/s (tg64)")
    ax.set_title("qwen2.5-1.5B decode speed vs context depth", fontsize=9)
    ax.set_ylim(0, max(vk) * 1.25)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-decode-vs-depth.png"), bbox_inches="tight")
    print("fig-decode-vs-depth.png")

# ---------------- fig 3: sgemm throughput ----------------
def fig_sgemm():
    ns = sorted(SGEMM); gf = [gflops(n, SGEMM[n]) for n in ns]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(ns, gf, "-o", color=VK, ms=5)
    for n, v in zip(ns, gf):
        ax.text(n, v * 1.05, f"{v/1000:.1f} TF", fontsize=7.5, ha="center")
    ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("matrix size N (SGEMM, N x N x N)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("native gfx1013 rocBLAS SGEMM, 20 iterations per size, all correct", fontsize=9)
    ax.set_ylim(0, max(gf) * 1.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-sgemm-curve.png"), bbox_inches="tight")
    print("fig-sgemm-curve.png")

# ---------------- fig 4: why prefill trails, by GEMM shape ----------------
def fig_gemm_shapes():
    labels = [x[0] for x in GEMM_SHAPES]
    vals = [x[1] / 1000.0 for x in GEMM_SHAPES]
    kinds = [x[2] for x in GEMM_SHAPES]
    colors = [VK if k == "square" else HIP for k in kinds]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.6, color=colors, edgecolor="black", linewidth=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.08, "%.1f" % v, ha="center", fontsize=8)

    # No prefill lines here. ROCm prefill on the model measured does not go
    # through rocBLAS at all (zero calls under ROCBLAS_LAYER=1), so drawing it
    # against these bars would imply a relationship that does not exist.
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("TFLOP/s")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("rocBLAS fp16 GEMM: layer shapes against square references", fontsize=9.5)
    fig.savefig(os.path.join(OUT, "fig-gemm-shapes.png"), bbox_inches="tight")
    print("fig-gemm-shapes.png")


if __name__ == "__main__":
    fig_backends(); fig_depth(); fig_sgemm(); fig_gemm_shapes()
