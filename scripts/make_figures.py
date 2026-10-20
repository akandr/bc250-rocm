#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Figures for the BC-250 ROCm README. Colours come from figstyle.py: ROCm red, Vulkan blue.

Data below is transcribed from bench-2026-08 logs (llama-bench tables,
sgemm_iter medians). Regenerate: python3 make_figures.py -> ../repo/figures/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as S

_d = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_d) == "scripts":            # repo/scripts/ -> repo/figures
    OUT = os.path.join(_d, "..", "figures")
else:                                            # validation dir -> ../repo/figures
    OUT = os.path.join(_d, "..", "repo", "figures")
os.makedirs(OUT, exist_ok=True)

VK = S.VULKAN
HIP = S.ROCM
S.apply()
OUT = S.out(OUT)

# ---------------- data (fill/verify from logs) ----------------
# model label -> (hip_tg, vk_tg, hip_pp512, vk_pp512); None = not measured/failed
# ROCm: fixed stack (integrated flag, KQV precision, RDNA1 macro), fa on,
# native gfx1013 rocBLAS, f32 cuBLAS compute type. Vulkan: same build, fa on.
# All rows perplexity-gated against Vulkan (2026-08-12 campaign). Not the same
# boot: the Vulkan rows were taken on 12 August and four of the five HIP rows on
# 13 August, thirteen hours apart, and whether the board stayed up between them
# is not recorded. The figure said "same boot" until 26 August, after the prose
# beside it had already been corrected.
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
# Fedora 44, native gfx1013 rocBLAS 7.1.1, GFLOP/s at a clock sampled and verified at 1500 MHz
# (logs/alu-rates-recheck-2026-09-25/rocblas_recheck.out).
#
# This used to plot logs/fedora44-benchmarks-2026-09-15/gpgpu/, which that directory's own README
# warns was taken with the governor oscillating: it reads 23.2 ms at 4096 and 187.55 at 8192, about
# 30 percent fast, and drew a Fedora 44 speedup that does not exist. The corrected run of the same
# probe is 30.1 and 237.9 ms, which is where Fedora 43 already sat.
SGEMM_F44_GF = {512: 1440.9, 1024: 2530.9, 2048: 2989.7, 4096: 4520.6, 8192: 4611.9}

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
                          va="center", fontsize=7.5, color=S.INK_SOFT)
        ax.set_yticks(y); ax.set_yticklabels(labels if ax is a1 else [""] * len(rows))
        ax.invert_yaxis(); ax.set_title(title, fontsize=9)
        if log: ax.set_xscale("log"); ax.set_xlim(10, max(vk_pp) * 4)
        else: ax.set_xlim(0, max(vk_tg) * 1.22)
    a1.legend(loc="center right", fontsize=7.5, frameon=False)
    fig.text(0.01, -0.04, "ROCm: llama.cpp master with the three gfx1013 fixes, flash attention on, "
             "native gfx1013 rocBLAS, f32 cuBLAS compute type.\nVulkan: same build, measured "
             "thirteen hours earlier; not the same boot. "
             "Every row passes a wikitext perplexity gate against Vulkan.", fontsize=7, color=S.NOTE)
    fig.suptitle("llama.cpp on the BC-250 at 40 CU: ROCm/HIP vs Vulkan (same build)",
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
    gf44 = [SGEMM_F44_GF[n] for n in ns]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(ns, gf44, "-o", color=S.ROCM, ms=5, label="Fedora 44, rocBLAS 7.1.1")
    ax.plot(ns, gf, ":s", color=S.GREY_MID, ms=4, label="Fedora 43, rocBLAS 6.4.2")
    ax.axhline(6520, color=S.GREY_MID, lw=1.0, ls="--")
    ax.text(8192, 6300, "measured fp32 ceiling 6520", fontsize=7, color=S.NOTE, ha="right", va="top")
    for n, v in zip(ns, gf44):
        ax.text(n, v + 330, f"{v/1000:.1f} TF", fontsize=7.5, ha="center", color=S.ROCM_DARK)
    ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("matrix size N (SGEMM, N x N x N)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("native gfx1013 rocBLAS SGEMM: the two releases measure the same", fontsize=9)
    ax.set_ylim(0, 7200); ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-sgemm-curve.png"), bbox_inches="tight")
    print("fig-sgemm-curve.png")

# ---------------- fig 4: why prefill trails, by GEMM shape ----------------
def fig_gemm_shapes():
    labels = [x[0] for x in GEMM_SHAPES]
    vals = [x[1] / 1000.0 for x in GEMM_SHAPES]
    kinds = [x[2] for x in GEMM_SHAPES]
    colors = [S.GREY_MID if k == "square" else S.ROCM for k in kinds]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, vals, 0.6, color=colors, edgecolor="none")
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



# ================= Fedora 44 / ROCm 7.1.1 (default from 2026-09-15) =================
# F44: logs/fedora44-benchmarks-2026-09-15/clock-corrected/, one test per llama-bench invocation (-p 512 -n 0 and
# -p 0 -n 64 separately), three alternated rounds of -r 3. F44_COMBINED: campaign/log, -p 512 -n 64 -r 3 in one
# invocation, used only against the Fedora 43 campaign, which was run the same way.
# ROCm with the native gfx1013 rocBLAS 7.1.1 and corrected comgr (no environment variables), Vulkan on
# Mesa 26.1.8, same llama.cpp tree, same boot. Fedora 43 columns are the MODELS dict above
# (12-13 August campaign; qwen3.8 27B from 17 August at tg128).
F44 = {  # label -> (hip_tg, vk_tg, hip_pp512, vk_pp512), medians of 9 samples, clocks pinned at 1500 MHz
    "qwen2.5 1.5B Q4_K_M": (117.49, 212.29, 792.88, 1849.1),
    "qwen3 8B Q8_0": (38.94, 39.07, 243.57, 394.77),
    "deepseek-r1 14B Q4_K_M": (21.38, 35.06, 94.45, 199.78),
    "qwen3 14B Q4_K_M": (21.8, 34.76, 96.78, 204.67),
    "qwen3.6 35B-A3B MoE IQ2_M": (34.29, 87.3, 289.66, 457.17),
    "qwen3.8 27B UD-IQ3_XXS": (7.85, 17.62, 69.43, 104.95),
}
# logs/fedora44-campaign-final-2026-09-18: the same split campaign on the four-patch build in its
# final form (RDNA1 flash-attention rows and the no-GQA dispatch check). Vulkan rows reproduce the
# 2026-09-15 session within 0.4 percent.
# This is the front page's headline figure, so it carries the same build and the same arm as the
# headline table: the thirteen patches as shipped, no runtime options, the as-shipped arm of
# logs/campaign-graphopt-2026-09-24 (medians.txt). It used to plot the arm with GGML_CUDA_GRAPH_OPT=1,
# an option that is off by default and computes wrong tokens without its fix.
F44_4PATCH = {  # decode ROCm, decode Vulkan, prefill ROCm, prefill Vulkan
    "qwen2.5 1.5B Q4_K_M": (196.7, 212.3, 1798.6, 1850.0),
    "qwen3 8B Q8_0": (38.5, 39.0, 409.4, 394.7),
    "deepseek-r1 14B Q4_K_M": (32.6, 35.1, 195.8, 199.8),
    "qwen3 14B Q4_K_M": (32.7, 34.7, 197.6, 204.8),
    "qwen3.6 35B-A3B MoE IQ2_M": (71.2, 86.9, 588.8, 457.0),
    "qwen3.8 27B UD-IQ3_XXS": (15.2, 17.6, 102.9, 105.0),
}
F44_COMBINED = {  # same order; -p 512 -n 64 in one invocation, as the Fedora 43 campaign was run
    "qwen2.5 1.5B Q4_K_M":        (146.42, 240.80, 995.10, 2417.90),
    "qwen3 8B Q8_0":              (40.52, 36.47, 306.81, 283.86),
    "deepseek-r1 14B Q4_K_M":     (21.21, 34.25, 122.56, 141.40),
    "qwen3 14B Q4_K_M":           (21.63, 33.94, 126.06, 143.55),
    "qwen3.6 35B-A3B MoE IQ2_M":  (42.52, 102.92, 372.05, 590.26),
    "qwen3.8 27B UD-IQ3_XXS":     (9.43, 19.20, 52.14, 73.57),
}
# logs/fedora44-benchmarks-2026-09-15/depth-clock-corrected/, qwen2.5-1.5B tg64 after d tokens,
# one depth per invocation, medians of three samples, clock pinned at 1500 MHz
# one depth per llama-bench invocation, -mmp 0 -r 3: (mean, min, median, max) for hip, vk
F44_DEPTH = {0: (117.71, 212.37), 4096: (108.9, 179.66), 8192: (99.95, 163.68), 16384: (87.45, 140.13), 24576: (76.21, 121.81), 30720: (70.24, 111.31)}
# logs/depth-thirteen-2026-09-22/log-perinvocation, section H: the same ladder on the thirteen-patch
# build, one llama-bench invocation per depth, three passes, with the Vulkan arm re-measured beside it
# (it returns 0.983 to 0.999 of the figures above, so the reference did not move).
# Three passes of logs/depth-graphopt-fix-2026-10-10/, all arms interleaved within each depth, medians:
# (ROCm as shipped, ROCm with GGML_CUDA_GRAPH_OPT=1 and its fix, Vulkan). This used to plot
# logs/depth-graphopt-2026-09-24/, whose option arm computes wrong tokens.
F44_DEPTH_GO = {0: (196.6, 210.9, 212.0), 4096: (176.3, 187.1, 179.1), 8192: (162.9, 171.9, 161.6),
                16384: (141.1, 147.5, 138.1), 24576: (122.9, 127.8, 121.0), 30720: (112.3, 116.3, 111.3)}
F44_DEPTH_13 = {0: (197.20, 212.24), 4096: (176.54, 179.28), 8192: (163.00, 160.92),
                16384: (141.30, 137.94), 24576: (123.15, 120.72), 30720: (112.61, 111.13)}
# logs/fedora44-validation-2026-09-15/probes: qwen2.5-1.5B, -r 3, same probe on both OSes
PROBES = [  # label, f43, f44 (both with the GPU clock pinned at 1500 MHz)
    ("pp512", 808.08, 790.92),
    ("pp512, fa off", 895.47, 896.23),
    ("pp2048, ubatch 2048", 746.13, 717.46),
    ("tg64", 113.75, 117.79),
    ("tg64, graphs off", 119.70, 119.90),
    ("tg64, fa off", 109.59, 113.20),
]
F43C = S.ROCM_PALE   # Fedora 43 ROCm series

def fig_f44_backends():
    # The ratio is what a reader comes for, so it is the axis: a bar to the right of 1 is ROCm ahead, to the
    # left Vulkan ahead, on one linear scale for both halves. The rates go in a column beside each panel.
    rows = list(F44_4PATCH.items()); labels = [r[0] for r in rows]
    y = np.arange(len(rows)); lo, hi, right = 0.70, 1.40, 1.80
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), sharey=True)
    for ax, ri, vi, title in ((axes[0], 2, 3, "prefill (pp512)"), (axes[1], 0, 1, "decode (tg64)")):
        ax.axvspan(lo, 1.0, color=S.VULKAN_BAND, zorder=0, lw=0)
        ax.axvspan(1.0, hi, color=S.ROCM_BAND, zorder=0, lw=0)
        ax.axvline(1.0, color=S.INK, lw=0.9, zorder=2)
        for yy, (name, v) in zip(y, rows):
            r = v[ri] / v[vi]
            col = S.ROCM if r >= 1 else S.VULKAN
            ax.barh(yy, r - 1.0, left=1.0, height=0.58, color=col, zorder=3)
            ax.text(r + (0.012 if r >= 1 else -0.012), yy, f"{r:.2f}x", ha="left" if r >= 1 else "right",
                    va="center", fontsize=8.5, fontweight="bold", color=col, zorder=4)
            ax.text(right - 0.01, yy, f"{v[ri]:.1f}  /  {v[vi]:.1f}", ha="right", va="center", fontsize=8,
                    color=S.INK)
        ax.text(right - 0.01, -0.72, "ROCm / Vulkan, tokens/s", ha="right", va="center", fontsize=7.5,
                color=S.GREY)
        ax.text(lo + 0.01, len(rows) - 0.35, "Vulkan faster", ha="left", va="center", fontsize=8,
                color=S.VULKAN, fontweight="bold")
        ax.text(hi - 0.01, len(rows) - 0.35, "ROCm faster", ha="right", va="center", fontsize=8,
                color=S.ROCM, fontweight="bold")
        ax.set_xlim(lo, right); ax.set_ylim(len(rows) - 0.05, -1.05)
        ax.set_xticks([0.8, 0.9, 1.0, 1.1, 1.2, 1.3]); ax.set_xticklabels(["0.8", "0.9", "1.0", "1.1", "1.2", "1.3"])
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.grid(axis="y", visible=False); ax.grid(axis="x", color=S.BG, linewidth=0.8)
        ax.spines["bottom"].set_bounds(lo, hi); ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels)
    fig.text(0.01, -0.04, "Fedora 44, kernel 7.2.5 with the bc250 amdgpu module, 40 CU, GPU clock capped at 1500 MHz. llama.cpp "
             "7ba604f with the thirteen gfx1013 patches and no runtime\noptions, both backends built from the same tree, "
             "flash attention on, empty context. Medians of eighteen samples over two campaigns "
             "(logs/campaign-graphopt-2026-09-24).", fontsize=7, color=S.NOTE)
    fig.suptitle("llama.cpp on the BC-250: ROCm against Vulkan", fontsize=10.5, y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-f44-rocm-vs-vulkan.png"), bbox_inches="tight")
    print("fig-f44-rocm-vs-vulkan.png")

def fig_f43_vs_f44():
    labels = list(F44.keys()); y = np.arange(len(labels)); h = 0.2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 0.9 + 0.8 * len(labels)))
    series = [  # name, source dict, index offset (0 hip, 1 vk), color, hatch
        ("ROCm, Fedora 43 (6.4.2)", MODELS, 0, F43C, ""), ("ROCm, Fedora 44 (7.1.1)", F44, 0, HIP, ""),
        ("Vulkan, Fedora 43 (Mesa 25.3.4)", MODELS, 1, S.VULKAN_PALE, ""), ("Vulkan, Fedora 44 (Mesa 26.1.8)", F44, 1, VK, ""),
    ]
    for ax, base, title, log in ((a1, 0, "decode (tokens/s)", False), (a2, 2, "prefill pp512 (tokens/s, log scale)", True)):
        for k, (name, src, off, col, hatch) in enumerate(series):
            vals = [src[l][base + off] for l in labels]
            yy = y + (k - 1.5) * h
            ax.barh(yy, vals, h, color=col, edgecolor=S.BG, linewidth=0.4, hatch=hatch, label=name)
            for v, yv in zip(vals, yy):
                ax.text(v * (1.1 if log else 1.02), yv, f"{v:g}", va="center", fontsize=6.3, color=S.INK_SOFT)
        ax.set_yticks(y); ax.set_yticklabels(labels if ax is a1 else [""] * len(labels)); ax.invert_yaxis()
        ax.set_title(title, fontsize=9)
        if log: ax.set_xscale("log"); ax.set_xlim(10, 6000)
        else: ax.set_xlim(0, 285)
    hs, ls = a1.get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.text(0.01, -0.13, "Same board, kernel, llama.cpp patches and GPU clock policy (1500 MHz). Fedora 43: August campaign, "
             "prefill and decode in one invocation\n(27B: 17 August, tg128); Fedora 44: 16 September, separate invocations, which "
             "raises large-model decode by about a quarter. Read the\ndecode bars as an upper bound on any difference.",
             fontsize=7, color=S.NOTE)
    fig.suptitle("Fedora 43 against Fedora 44 at the same GPU clock: no speed difference", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-f43-vs-f44.png"), bbox_inches="tight")
    print("fig-f43-vs-f44.png")

def fig_probes():
    labels = [p[0] for p in PROBES]; a = [p[1] for p in PROBES]; b = [p[2] for p in PROBES]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - w/2, a, w, color=F43C, edgecolor="none", label="Fedora 43, ROCm 6.4.2")
    ax.bar(x + w/2, b, w, color=HIP, edgecolor="none", label="Fedora 44, ROCm 7.1.1")
    for i in range(len(labels)):
        ax.text(x[i] + w/2, b[i] * 1.16, f"{(b[i]/a[i]-1)*100:+.0f}%", ha="center", fontsize=7.5, fontweight="bold")
    # A log axis over this range labels only 10^2 and leaves the tall bars running off the top with
    # nothing to read them against, so the decades are ticked explicitly and every bar carries its value.
    ax.set_yscale("log")
    ax.set_ylim(80, max(max(a), max(b)) * 1.6)
    ax.set_yticks([100, 200, 400, 800, 1600])
    ax.set_yticklabels(["100", "200", "400", "800", "1600"])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5, rotation=15)
    ax.set_ylabel("tokens/s (log scale)")
    for i in range(len(labels)):
        ax.text(x[i] - w/2, a[i] * 1.02, f"{a[i]:.0f}", ha="center", fontsize=6.8, color=S.NOTE)
        ax.text(x[i] + w/2, b[i] * 1.02, f"{b[i]:.0f}", ha="center", fontsize=6.8, color=S.NOTE)
    ax.set_title("qwen2.5-1.5B, same probes on both systems, clock capped at 1500 MHz", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-f44-speed-probes.png"), bbox_inches="tight")
    print("fig-f44-speed-probes.png")

def fig_f44_depth():
    ds = sorted(F44_DEPTH)
    hip = [F44_DEPTH[d][0] for d in ds]; vk = [F44_DEPTH[d][1] for d in ds]
    # All three of these come from ONE run, logs/depth-graphopt-fix-2026-10-10/, arms interleaved within
    # each depth. A figure that compared a new ROCm line against an old Vulkan line would decide where
    # the two cross by which run it happened to use, which this repository has already done once.
    hip13 = [F44_DEPTH_GO[d][0] for d in ds]
    hipopt = [F44_DEPTH_GO[d][1] for d in ds]
    vk13 = [F44_DEPTH_GO[d][2] for d in ds]
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(ds, vk13, "-o", color=VK, ms=5, label="Vulkan, same run")
    ax.plot(ds, hipopt, "-D", color=HIP, ms=5, label="ROCm, GGML_CUDA_GRAPH_OPT=1 with its fix")
    ax.plot(ds, hip13, "--D", color=S.ROCM_MID, ms=4, label="ROCm, as shipped")
    ax.plot(ds, hip, "--s", color=S.GREY, ms=4, label="ROCm, three patches")
    od = sorted(DEPTH)
    ax.plot(od, [DEPTH[d][1] for d in od], ":o", color=S.VULKAN_MID, ms=3, label="Vulkan, Fedora 43")
    ax.plot(od, [DEPTH[d][0] for d in od], ":s", color=S.GREY_MID, ms=3, label="ROCm, Fedora 43")
    for d, v in zip(ds, hip): ax.text(d, v - 12, f"{v:.0f}", fontsize=7, ha="center", color=S.NOTE)
    for d, v in zip(ds, hipopt): ax.text(d, v + 7, f"{v:.0f}", fontsize=7, ha="center", color=S.INK_SOFT)
    for d, v in zip(ds, vk13): ax.text(d, v - 14, f"{v:.0f}", fontsize=7, ha="center", color=VK)
    ax.set_xlabel("context depth before generation (tokens)"); ax.set_ylabel("decode tokens/s (tg64)")
    ax.set_title("qwen2.5-1.5B decode vs depth: level with Vulkan from 4096 as shipped, ahead with the option",
                 fontsize=9)
    ax.set_ylim(0, 255); ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-f44-decode-vs-depth.png"), bbox_inches="tight")
    print("fig-f44-decode-vs-depth.png")


# ================= flash attention at depth (logs/flash-attention-tradeoff-2026-09-17) =================
# qwen3-8B Q8_0, one test per llama-bench invocation, -r 2, clock pinned. Medians.
# prefill is pp2048 at the given existing context depth; decode is tg64 at the same depth.
FA_PP = {0:    {"hip_on": 198.22, "hip_off": 250.43, "vk_on": 367.46, "vk_off": 332.35},
         4096: {"hip_on": 98.08,  "hip_off": 186.09, "vk_on": 216.92, "vk_off": 255.30},
         8192: {"hip_on": 65.51,  "hip_off": 133.26, "vk_on": 143.98, "vk_off": 220.35}}
FA_TG = {4096: {"hip_on": 32.73, "hip_off": 18.59, "vk_on": 35.72, "vk_off": 30.17},
         8192: {"hip_on": 28.32, "hip_off": 12.47, "vk_on": 33.24, "vk_off": 23.77}}
# logs/vulkan-fa-staging-2026-09-17: same model, Vulkan with llama.cpp PR #28507 (fa on)
VK_PATCHED_PP = {0: 377.20, 4096: 293.62, 8192: 236.13}

def fig_fa_tradeoff():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, data, depths, title in (
        (a1, FA_PP, [0, 4096, 8192], "prefill: pp2048 at that depth"),
        (a2, FA_TG, [4096, 8192], "decode: tg64 at that depth"),
    ):
        labels, on, off = [], [], []
        for be, bename in (("hip", "ROCm"), ("vk", "Vulkan")):
            for d in depths:
                labels.append(f"{bename}\nd{d}")
                on.append(data[d][be + "_on"]); off.append(data[d][be + "_off"])
        x = np.arange(len(labels)); w = 0.38
        hue = [(S.ROCM, S.ROCM_PALE) if l.startswith("ROCm") else (S.VULKAN, S.VULKAN_PALE) for l in labels]
        ax.bar(x - w/2, on, w, color=[h[0] for h in hue], edgecolor="none")
        ax.bar(x + w/2, off, w, color=[h[1] for h in hue], edgecolor="none")
        for i in range(len(labels)):
            hi, lo = max(on[i], off[i]), min(on[i], off[i])
            ax.text(x[i], hi * 1.03, f"{hi/lo:.1f}x", ha="center", fontsize=7.5, color=S.INK_SOFT)
            # Contrast follows the bar, not the value: keying off the number left every label in the
            # decode panel dark, including the ones sitting on the near-black bars.
            ax.text(x[i] - w/2, on[i] * 0.5, f"{on[i]:.0f}", ha="center", fontsize=7, color=S.ON_BAR)
            ax.text(x[i] + w/2, off[i] * 0.5, f"{off[i]:.0f}", ha="center", fontsize=7, color=S.INK)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title, fontsize=9); ax.set_ylabel("tokens/s")
        ax.set_ylim(0, max(max(on), max(off)) * 1.2)
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color=S.GREY, label="flash attention on (%s)" % ("bright" if S.DARK else "dark")),
                       Patch(color=S.GREY_PALE, label="flash attention off (%s)" % ("dim" if S.DARK else "light"))],
              fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Flash attention before the RDNA1 patch: faster prefill without it, faster decode with it", fontsize=10)
    fig.text(0.01, -0.06, "qwen3-8B Q8_0, Fedora 44, kernel 7.2.5, clock capped at 1500 MHz. Labels above each pair are the ratio.\n"
             "Turning it off also costs KV-cache memory, so it reduces the context that fits.", fontsize=7, color=S.NOTE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-flash-attention-tradeoff.png"), bbox_inches="tight")
    print("fig-flash-attention-tradeoff.png")

# logs/rdna1-fattn-spill-2026-09-17/depth-final.log: the same 8B prefill against depth on the final
# four-patch build, -fa on, medians of two passes. Empty until measured.
HIP_FINAL_PP = {0: 266.4, 4096: 219.7, 8192: 184.5}
# logs/depth-thirteen-2026-09-22/log-followup, section D: the thirteen-patch build, one llama-bench
# invocation per depth, three passes. The Vulkan series above were re-measured in the same run and
# return 367.02, 217.46 and 143.76 against the 366.82, 217.01 and 143.51 plotted, so they stand.
HIP_13_PP = {0: 394.42, 4096: 300.52, 8192: 214.55}

def fig_prefill_depth():
    ds = sorted(FA_PP)
    series = [("Vulkan, fa off", [FA_PP[d]["vk_off"] for d in ds], S.VULKAN_MID, "-o", ""),
              ("Vulkan with PR #28507, fa on", [VK_PATCHED_PP[d] for d in ds], S.VULKAN_DARK, "--^", ""),
              ("Vulkan, fa on", [FA_PP[d]["vk_on"] for d in ds], S.VULKAN, ":o", "")]
    if HIP_13_PP:
        series.append(("ROCm, fa on, thirteen patches", [HIP_13_PP[d] for d in ds], S.ROCM, "-D", ""))
    if HIP_FINAL_PP:
        series.append(("ROCm, fa on, four patches", [HIP_FINAL_PP[d] for d in ds], S.ROCM_MID, "--D", ""))
    series += [("ROCm, fa off", [FA_PP[d]["hip_off"] for d in ds], S.AMBER, "-s", ""),
               ("ROCm, fa on, three patches" if HIP_FINAL_PP else "ROCm, fa on",
                [FA_PP[d]["hip_on"] for d in ds], S.ROCM_DARK, ":s", "")]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, vals, col, style, _ in series:
        ax.plot(ds, vals, style, color=col, ms=5, label=name, linewidth=1.6)
    for name, vals, col, style, _ in series:
        # 8192 is crowded: 236, 220, 215, 184, 144, 133 and 66 within one axis height.
        dy = {"Vulkan with PR #28507, fa on": 10, "Vulkan, fa off": 2,
              "ROCm, fa on, thirteen patches": -11, "ROCm, fa on, four patches": -4,
              "Vulkan, fa on": 7, "ROCm, fa off": -8}.get(name, 0)
        ax.text(ds[-1] + 400, vals[-1] + dy, f"{vals[-1]:.0f}", fontsize=7.5, va="center", color=col,
                fontweight="bold")
    ax.set_xlabel("tokens of context already present")
    ax.set_ylabel("prefill tokens/s (pp2048)")
    ax.set_title("Prefill against context: where the flash-attention patch matters most" if HIP_13_PP
                 else "Prefill falls with context, and flash attention makes it worse", fontsize=9.5)
    ax.set_xticks(ds); ax.set_xlim(-500, ds[-1] + 1600); ax.set_ylim(0, 430)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-prefill-vs-depth.png"), bbox_inches="tight")
    print("fig-prefill-vs-depth.png")


# ================= prefill against depth to 24576 (logs/prefill-deep-2026-10-12) =================
# The same 8B prefill ladder carried to 16384 and 24576 cached tokens, three arms interleaved within each
# depth, medians of three passes: ROCm build-hip-pkf16 (thirteen patches), Vulkan build-vk-f44, and the
# vulkan-fa-staging build of master bfdc321 carrying PR #28507. At 24576 every arm ran part of the time at
# 1000 MHz (the shader clock sampled during each run: 25 to 40 percent of the samples, ROCm the most), so
# the last ROCm/Vulkan ratio is a floor, not a measurement at 1500 MHz. The 28672 point (30720 in context,
# the deepest pp2048 under the 8B's 32768 ROCm limit) is logs/ladders-rest-2026-10-17/a: -r 1 after three
# idle minutes, medians of three; the governor overheated during one ROCm run there too.
PP24 = {
    "ROCm, thirteen patches": {0: 390.9, 4096: 302.2, 8192: 246.5, 16384: 171.8, 24576: 134.6, 28672: 122.7},
    "Vulkan": {0: 367.4, 4096: 217.9, 8192: 143.2, 16384: 84.1, 24576: 58.9, 28672: 51.1},
    "Vulkan with PR #28507": {0: 377.2, 4096: 294.1, 8192: 242.4, 16384: 176.5, 24576: 135.3, 28672: 123.0},
}

def fig_prefill_depth_24k():
    ds = sorted(PP24["Vulkan"])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvspan(20480, 30300, color=S.GREY_PALE, alpha=0.45, lw=0, zorder=0)
    ax.text(26624, 8, "all three arms\nthrottle here", ha="center", va="bottom", fontsize=7, color=S.GREY)
    styles = {"ROCm, thirteen patches": (S.ROCM, "-D"), "Vulkan": (S.VULKAN, "-o"),
              "Vulkan with PR #28507": (S.VULKAN_DARK, "--^")}
    for name, (col, style) in styles.items():
        vals = [PP24[name][d] for d in ds]
        ax.plot(ds, vals, style, color=col, ms=5, linewidth=1.6, label=name, zorder=3)
        ax.text(ds[-1] + 500, vals[-1] + (6 if "PR" in name else -6 if "ROCm" in name else 0),
                f"{vals[-1]:.0f}", fontsize=7.5, va="center", color=col, fontweight="bold")
    for d in ds:  # ROCm over stock Vulkan, written above the ROCm point
        r = PP24["ROCm, thirteen patches"][d] / PP24["Vulkan"][d]
        ax.text(d, PP24["ROCm, thirteen patches"][d] + 14, f"{r:.2f}x", ha="center", fontsize=7.5,
                color=S.ROCM, fontweight="bold")
    ax.set_xlabel("tokens of context already present")
    ax.set_ylabel("prefill tokens/s (pp2048)")
    ax.set_title("Prefill against context to 28k: ROCm over stock Vulkan grows with depth, level with #28507",
                 fontsize=9.5)
    ax.set_xticks(ds); ax.set_xticklabels(["0", "4k", "8k", "16k", "24k", "28k"])
    ax.set_xlim(-600, ds[-1] + 2600); ax.set_ylim(0, 430)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.text(0.01, -0.04, "qwen3-8B Q8_0, -p 2048 -n 0, -fa on, one llama-bench invocation per point, medians of three passes, 2026-10-12 (28k: 2026-10-17). "
             "Labels above the ROCm points are ROCm / Vulkan.", fontsize=7, color=S.NOTE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-prefill-vs-depth-24k.png"), bbox_inches="tight")
    print("fig-prefill-vs-depth-24k.png")


# ================= decode against depth on the 8B (logs/ladders-2026-10-15) =================
# qwen3-8B Q8_0 tg64 behind 0 to 24576 cached tokens, three arms interleaved per depth, medians of three
# passes, the no-depth-cache llama-bench. The 16384 and 24576 ROCm points of the main run throttled (30 to
# 40 percent of the clock samples at 1000 MHz); the cold points are phase 2, -r 1 after three idle minutes.
DEC8 = {
    "ROCm, thirteen patches, as shipped": {0: 38.43, 4096: 36.13, 8192: 34.03, 16384: 29.62, 24576: 27.53},
    "ROCm, GGML_CUDA_GRAPH_OPT=1 with its fix": {0: 39.85, 4096: 37.08, 8192: 34.72, 16384: 29.27, 24576: 27.43},
    "Vulkan": {0: 39.06, 4096: 35.73, 8192: 33.22, 16384: 29.25, 24576: 26.19},
}
DEC8_COLD = {   # phase 2: -r 1 after three idle minutes, medians of three, no throttling; 28672 is
                # logs/ladders-rest-2026-10-17/d (2026-10-18), same method, the last rung the no-mmap 8B holds on ROCm
    "ROCm, thirteen patches, as shipped": {16384: 30.71, 24576: 28.00, 28672: 26.83},
    "ROCm, GGML_CUDA_GRAPH_OPT=1 with its fix": {16384: 31.44, 24576: 28.54, 28672: 27.34},
    "Vulkan": {16384: 29.32, 24576: 26.14, 28672: 25.02},
}

def fig_decode_depth_8b():
    ds = sorted(DEC8["Vulkan"])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    styles = {"ROCm, thirteen patches, as shipped": (S.ROCM, "-D"),
              "ROCm, GGML_CUDA_GRAPH_OPT=1 with its fix": (S.ROCM_DARK, "--D"), "Vulkan": (S.VULKAN, "-o")}
    for name, (col, style) in styles.items():
        vals = [DEC8[name][d] for d in ds]
        ax.plot(ds, vals, style, color=col, ms=5, linewidth=1.6, label=name, zorder=3)
        ax.text(ds[-1] + 400, vals[-1] + (0.5 if "fix" in name else -0.5 if "shipped" in name else 0),
                f"{vals[-1]:.1f}", fontsize=7.5, va="center", color=col, fontweight="bold")
    for name, cold in DEC8_COLD.items():
        col = styles[name][0]
        ax.plot(sorted(cold), [cold[d] for d in sorted(cold)], "x", color=col, ms=6, mew=1.4, zorder=4,
                label="phase 2, cold -r 1 (x)" if "Vulkan" in name else None)
        if 28672 in cold:
            ax.text(28672 + 450, cold[28672] + (0.45 if "fix" in name else -0.45 if "shipped" in name else 0),
                    f"{cold[28672]:.1f}", fontsize=7.5, va="center", color=col, fontweight="bold")
    ax.axvspan(14000, 30300, color=S.GREY_PALE, alpha=0.45, lw=0, zorder=0)
    ax.text(20480, 36.8, "ROCm throttles here\nin the three-pass run", ha="center", va="top", fontsize=7, color=S.GREY)
    for d in ds:
        r = DEC8["ROCm, thirteen patches, as shipped"][d] / DEC8["Vulkan"][d]
        ax.text(d, 24.9, f"{r:.2f}x", ha="center", va="bottom", fontsize=7.5, color=S.ROCM, fontweight="bold")
    ax.text(-400, 23.85, "this row: ROCm as shipped / Vulkan, three-pass run", ha="left", va="bottom", fontsize=6.5, color=S.ROCM)
    ax.set_xlabel("tokens of context already present"); ax.set_ylabel("decode tokens/s (tg64)")
    ax.set_title("Decode against context on the 8B: ROCm ahead from 4k, where on the 1.5B it is level", fontsize=9.5)
    ax.set_xticks(ds + [28672]); ax.set_xticklabels(["0", "4k", "8k", "16k", "24k", "28k"])
    ax.set_xlim(-600, 28672 + 2400); ax.set_ylim(23.6, 43)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.text(0.01, -0.07, "qwen3-8B Q8_0, -p 0 -n 64, one llama-bench invocation per point (no-depth-cache build), medians of three passes, 2026-10-15.\n"
             "Crosses: phase 2, the same arms cold with -r 1, where the three-pass run throttled; 28k (2026-10-18) the same way. ROCm does not hold 30k without mmap.", fontsize=7, color=S.NOTE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-decode-vs-depth-8b.png"), bbox_inches="tight")
    print("fig-decode-vs-depth-8b.png")


# ================= decode against depth on all six models (logs/ladders-rest-2026-10-17) =================
# ROCm as shipped over Vulkan, tg64 behind 0 to 24576 cached tokens. The 1.5B is logs/depth-graphopt-fix-2026-10-10,
# the 8B is logs/ladders-2026-10-15 (its 16384 and 24576 the cold phase-2 points), the other four are
# logs/ladders-rest-2026-10-17: one no-depth-cache llama-bench invocation per point, -r 1, arms interleaved,
# medians of three (four where a point was measured in both attempts). deepseek-14B does not fit 24576 on
# ROCm; the 27B's 8192 and 16384 are phase C2, -r 1 after five idle minutes, because the 90-second rests of
# the main run let the governor throttle its long fills.
DEC6 = {
    "qwen2.5-1.5B Q4_K_M": {0: 196.6 / 212.0, 4096: 176.3 / 179.1, 8192: 162.9 / 161.6, 16384: 141.1 / 138.1, 24576: 122.9 / 121.0},
    "qwen3-8B Q8_0": {0: 38.43 / 39.06, 4096: 36.13 / 35.73, 8192: 34.03 / 33.22, 16384: 30.71 / 29.32, 24576: 28.00 / 26.14, 28672: 26.83 / 25.02},
    "deepseek-r1-14B Q4_K_M": {0: 32.6 / 35.1, 4096: 26.5 / 31.0, 8192: 22.4 / 28.1, 16384: 15.0 / 23.7},
    "qwen3-14B Q4_K_M": {0: 32.7 / 34.7, 4096: 27.3 / 31.4, 8192: 23.7 / 28.7, 16384: 16.5 / 25.0, 24576: 12.2 / 21.8},
    "qwen3.6-35B-A3B MoE IQ2_M": {0: 71.2 / 86.9, 4096: 69.0 / 83.9, 8192: 67.0 / 81.5, 16384: 63.5 / 78.1, 24576: 60.2 / 73.9},
    "qwen3.8-27B UD-IQ3_XXS": {0: 15.2 / 17.6, 4096: 14.9 / 17.3, 8192: 14.7 / 16.4, 16384: 10.8 / 16.5},   # 8192 and 16384 pool B and C2, six runs each
}

def fig_decode_depth_six():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhspan(1.0, 1.15, color=S.ROCM, alpha=S.BAND_ALPHA, lw=0, zorder=0)
    ax.axhspan(0.42, 1.0, color=S.VULKAN, alpha=S.BAND_ALPHA, lw=0, zorder=0)
    ax.axhline(1.0, color=S.GREY, lw=0.8, zorder=1)
    styles = {"qwen2.5-1.5B Q4_K_M": (S.PURPLE, "-o"), "qwen3-8B Q8_0": (S.ROCM, "-D"),
              "deepseek-r1-14B Q4_K_M": (S.BROWN, "-s"), "qwen3-14B Q4_K_M": (S.ORANGE, "--s"),
              "qwen3.6-35B-A3B MoE IQ2_M": (S.GREEN, "-^"),
              "qwen3.8-27B UD-IQ3_XXS": (S.VULKAN_DARK, "-v")}
    for name, (col, style) in styles.items():
        pts = DEC6[name]; ds = sorted(pts)
        ax.plot(ds, [pts[d] for d in ds], style, color=col, ms=4.5, linewidth=1.5, label=name, zorder=3)
        dy = 0.022 if "27B" in name else -0.012 if "deepseek" in name else 0
        ax.text(ds[-1] + 400, pts[ds[-1]] + dy, f"{pts[ds[-1]]:.2f}", fontsize=7.5, va="center", color=col, fontweight="bold")
    ax.text(31000, 1.13, "ROCm faster", color=S.ROCM, fontsize=8, alpha=0.7, va="top", ha="right")
    ax.text(31000, 0.435, "Vulkan faster", color=S.VULKAN, fontsize=8, alpha=0.7, va="bottom", ha="right")
    ax.set_xlabel("tokens of context already present"); ax.set_ylabel("decode speed, ROCm / Vulkan (tg64)")
    ax.set_title("Decode against context on six models: ahead on the 8B, level on the 1.5B, the dense big ones fall away",
                 fontsize=9.5)
    ax.set_xticks([0, 4096, 8192, 16384, 24576, 28672]); ax.set_xticklabels(["0", "4k", "8k", "16k", "24k", "28k"])
    ax.set_xlim(-600, 28672 + 3200); ax.set_ylim(0.42, 1.15)
    ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0.0, 0.0))
    fig.text(0.01, -0.07, "ROCm thirteen patches as shipped / Vulkan, -p 0 -n 64, one no-depth-cache llama-bench invocation per point, -r 1, medians of three.\n"
             "1.5B: 2026-10-10, 8B: 2026-10-15 (16k and 24k cold), the other four: 2026-10-17. deepseek-14B does not fit 24k on ROCm, the 27B stops at 16k, the 8B at 28k.", fontsize=7, color=S.NOTE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-decode-vs-depth-six.png"), bbox_inches="tight")
    print("fig-decode-vs-depth-six.png")


# ================= eight-hour soak (logs/soak-thirteen-2026-09-22) =================
# 152 rounds rotating four models, 38 each: pp512/tg64, a six-chunk perplexity gate against the value
# measured when that model's patch landed, and an allocation-churn sweep every fourth round. The four
# models differ by a factor of seventeen in prefill rate, so each series is plotted against its own
# median: the question this figure answers is whether anything drifts, not how fast the models are.
# (label, round indices, pp512, tg64)
SOAK13 = [
    ("qwen2.5-1.5B", [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 129, 133, 137, 141, 145, 149], [1779.56, 1782.81, 1784.18, 1780.2, 1779.69, 1781.1, 1783.91, 1782.28, 1780.14, 1781.16, 1778.74, 1782.12, 1780.64, 1779.97, 1780.62, 1785.08, 1781.26, 1779.49, 1782.86, 1781.96, 1781.29, 1782.36, 1781.3, 1781.51, 1779.02, 1782.31, 1780.75, 1783.34, 1782.75, 1781.57, 1782.22, 1782.32, 1783.45, 1784.88, 1781.03, 1782.36, 1781.7, 1785.55], [194.82, 196.01, 197.03, 197.06, 197.46, 196.24, 197.18, 194.57, 194.73, 197.58, 196.74, 197.1, 196.94, 196.13, 196.75, 197.13, 196.8, 196.94, 197.45, 196.98, 196.59, 197.1, 197.22, 195.36, 196.67, 196.94, 196.27, 197.12, 197.51, 197.4, 196.89, 196.99, 197.04, 197.34, 197.42, 196.9, 197.02, 197.13]),
    ("qwen3-8B Q8_0", [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152], [414.58, 409.05, 409.56, 409.54, 415.24, 409.26, 410.82, 409.38, 409.78, 409.42, 414.98, 414.94, 415.33, 414.81, 413.55, 414.85, 413.88, 414.67, 415.16, 414.83, 414.83, 414.84, 414.84, 415.09, 414.62, 414.64, 415.02, 415.04, 415.1, 414.84, 414.98, 414.88, 415.05, 411.67, 415.01, 409.34, 409.91, 409.12], [38.72, 38.74, 38.75, 38.74, 38.76, 38.76, 38.76, 38.73, 38.74, 38.69, 38.72, 38.72, 38.73, 38.76, 38.73, 38.76, 38.73, 38.72, 38.74, 38.74, 38.77, 38.63, 38.75, 38.76, 38.75, 38.74, 38.76, 38.78, 38.76, 38.76, 38.78, 38.35, 38.7, 38.71, 38.77, 38.74, 38.72, 38.72]),
    ("qwen3.6-35B MoE", [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151], [595.08, 595.31, 595.2, 593.64, 595.02, 594.7, 593.21, 595.02, 592.98, 593.18, 590.01, 589.86, 589.84, 594.89, 589.75, 591.56, 595.02, 595.27, 594.45, 595.29, 594.4, 595.16, 595.36, 593.99, 595.64, 595.01, 593.04, 590.56, 590.27, 594.96, 595.19, 590.65, 591.81, 595.27, 595.37, 594.17, 595.04, 594.16], [71.84, 71.7, 71.81, 71.49, 71.64, 71.74, 71.87, 71.73, 71.76, 71.88, 71.95, 71.51, 71.26, 71.89, 71.85, 71.87, 71.87, 72.03, 71.66, 71.7, 71.87, 71.68, 71.69, 71.92, 71.55, 71.62, 71.42, 71.53, 71.94, 71.87, 71.87, 71.88, 71.95, 71.45, 71.97, 71.85, 71.52, 71.57]),
    ("qwen3.8-27B", [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126, 130, 134, 138, 142, 146, 150], [102.98, 103.37, 102.96, 103.38, 103.98, 105.16, 104.71, 103.58, 104.72, 104.56, 103.2, 103.87, 104.77, 105.06, 103.49, 104.94, 104.92, 104.08, 104.21, 104.94, 103.34, 104.13, 104.61, 104.37, 104.95, 104.48, 104.24, 104.16, 104.81, 104.99, 104.95, 104.71, 102.87, 102.78, 104.38, 104.56, 104.76, 104.13], [15.15, 15.17, 15.18, 15.22, 15.18, 15.21, 15.16, 15.17, 15.18, 15.16, 15.19, 15.21, 15.17, 15.22, 15.2, 15.2, 15.21, 15.17, 15.18, 15.21, 15.18, 15.19, 15.2, 15.18, 15.19, 15.17, 15.16, 15.2, 15.2, 15.2, 15.21, 15.18, 15.16, 15.2, 15.13, 15.18, 15.2, 15.18]),
]
SOAK13_GATES = [("qwen2.5-1.5B", "10.2088"), ("qwen3-8B Q8_0", "9.4017"),
                ("qwen3.6-35B MoE", "6.2265"), ("qwen3.8-27B", "6.2737")]

def fig_soak():
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.4), sharex=True)
    cols = [S.TEAL, S.ROCM, S.VIOLET, S.AMBER]
    marks = ["o", "s", "^", "D"]
    for ax, idx, name in ((axes[0], 2, "prefill pp512"), (axes[1], 3, "decode tg64")):
        for (label, rounds, pp, tg), col, mk in zip(SOAK13, cols, marks):
            ys = (pp, tg)[idx - 2]
            med = sorted(ys)[len(ys) // 2]
            ax.plot(rounds, [y / med for y in ys], "-", marker=mk, ms=2.6, linewidth=1.0,
                    color=col, label=label)
        ax.axhline(1.0, color=S.GREY_MID, linestyle="--", linewidth=0.8, zorder=0)
        ax.set_ylim(0.955, 1.045)
        ax.set_ylabel(name + "\n(of each model's median)", fontsize=8)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7.5, frameon=False, ncol=4, loc="upper center")
    axes[1].set_xlabel("round (each round: one model's pp512 and tg64, then its perplexity gate; "
                       "churn sweep every fourth)", fontsize=8)
    axes[0].set_title("Eight hours, 152 rounds, four models: every gate bit-identical, no faults",
                      fontsize=9.5)
    gates = ", ".join("%s %s" % g for g in SOAK13_GATES)
    fig.text(0.01, -0.035,
             "Fedora 44, kernel 7.2.5, amdgpu.gpu_recovery=0, clock capped at 1500 MHz, thirteen patches. "
             "Each model's gate returned one value\nacross all 38 of its rounds (" + gates +
             ");\n38 of 38 allocation-churn sweeps passed; the kernel logged no GPU fault. "
             "Spreads 0.4 to 2.3 percent.", fontsize=7, color=S.NOTE, va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-soak-stability.png"), bbox_inches="tight")
    print("fig-soak-stability.png")

# ========== per-op HIP vs Vulkan (logs/op-perf-hip-vs-vulkan-2026-09-17) ==========
# Median of three passes, us/run from test-backend-ops perf replaying the real pp2048
# graph. Label, HIP us, Vulkan us. Matmuls first, then the rest of the prefill graph.
OPS_MM = [
    ("FLASH_ATTN_EXT", 114650, 16260),
    ("ffn_down q6_K",   22688, 11535),
    ("ffn_down q4_K",   20654, 10385),
    ("ffn_up/gate q4_K", 19503, 11704),
    ("KQV f16",           8208,  5805),
    ("KQ f16",            8087,  4399),
    ("attn qkv/o q4_K",   3532,  1790),
]
OPS_OTHER = [
    ("SOFT_MAX",  3371.8, 3156.5),
    ("SWIGLU",     568.3,  580.2),
    ("ROPE Q",     199.4,  121.6),
    ("RMS_NORM",   126.1,   97.7),
    ("ADD bcast",   92.9,  210.9),
    ("MUL bcast",   92.8,  210.7),
]

def fig_op_perf():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4),
                             gridspec_kw={"width_ratios": [1, 1]})
    for ax, data, title in (
            (axes[0], OPS_MM, "Attention and matmuls carry the whole deficit"),
            (axes[1], OPS_OTHER, "The rest of the graph: mixed, median 0.75x")):
        labels = [d[0] for d in data]
        y = np.arange(len(labels))[::-1]
        ratio = [d[1] / d[2] for d in data]
        colors = [S.VULKAN if r > 1 else S.ROCM for r in ratio]
        ax.barh(y, [min(r, 2.45) for r in ratio], height=0.6, color=colors, edgecolor="none")
        ax.axvline(1.0, color=S.NOTE, linewidth=1)
        for yy, r in zip(y, ratio):
            clipped = r > 2.45
            ax.text(min(r, 2.45) - (0.06 if clipped else -0.04), yy, f"{r:.2f}x",
                    va="center", ha="right" if clipped else "left",
                    fontsize=7.5, color=S.ON_BAR if clipped else S.INK_SOFT)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 2.45); ax.set_xlabel("ROCm time / Vulkan time")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.09,
             "Replaying the real pp2048 graph of the 1.5B through test-backend-ops on each backend, median of the passes.\n"
             "Bars right of 1 are slower on ROCm; the flash attention bar runs off the axis at 7.05x and is labelled inside.\n"
             "With flash attention off the prefill graph sums to 1.74, against 1.86 end to end. Decode-shaped attention ties.",
             fontsize=7, color=S.NOTE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-op-perf-hip-vs-vulkan.png"), bbox_inches="tight")
    print("fig-op-perf-hip-vs-vulkan.png")


# ============ RDNA1 flash attention spill (logs/rdna1-fattn-spill-2026-09-17) ============
# hipcc -Rpass-analysis=kernel-resource-usage on flash_attn_tile<128,128,8,8,false>.
# arch label -> (VGPRs, VGPRs spilled, scratch bytes/lane)
SPILL = [
    ("gfx1010\nRDNA1", 256, 569, 2280),
    ("gfx1013\nRDNA1", 256, 569, 2280),
    ("gfx1030\nRDNA2", 215, 0, 0),
    ("gfx1100\nRDNA3", 125, 0, 0),
    ("gfx1013\npatched", 211, 0, 0),
]
# qwen3-8B Q8_0 pp2048 against depth (logs/rdna1-fattn-spill-2026-09-17/depth-sweep.log), three
# passes, builds interleaved at every point, medians. The -fa off arms run identical code in both
# builds and are pooled (six values); they scatter 7 to 10 percent at depth because that arm always
# follows a heavy -fa on run, so the pooled median is the honest figure and its spread is quoted.
# depth -> (fa off pooled, unpatched fa on, patched fa on)
SPILL_DEPTH = {0: (243.47, 194.27, 256.74), 4096: (165.44, 97.32, 195.53), 8192: (117.34, 64.93, 157.41)}

# Per head size: flash-attention prefill op time and end-to-end pp2048, before and after the
# RDNA1 rows (logs/rdna1-fattn-spill-2026-09-17: bench.log, bighead-ab.log). ms per op, tokens/s.
HEADS = [  # label, op before, op after, pp2048 before, pp2048 after (final rows, ab14b.log for the last four)
    ("qwen2.5-1.5B\nD=128",     114.7, 26.3, 634.2, 888.8),
    ("qwen3-8B\nD=128",          None,  None, 185.6, 264.1),
    ("deepseek-14B\nD=128, no GQA", None, None, 83.6, 92.6),
    ("qwen3-14B\nD=128, no GQA",    None, None, 87.6, 95.2),
    ("qwen3.6-35B MoE\nD=256",  338.6, 97.0, 262.3, 288.9),
    ("qwen3.8-27B\nD=256",      507.3, 149.4, 61.3, 65.0),
]

def fig_heads():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4), gridspec_kw={"width_ratios": [1, 1.7]})
    ax = axes[0]
    rows = [h for h in HEADS if h[1] is not None]
    x = np.arange(len(rows)); w = 0.36
    ax.bar(x - w/2, [h[1] for h in rows], w, color=S.ROCM_PALE, label="before")
    ax.bar(x + w/2, [h[2] for h in rows], w, color=S.ROCM, label="after")
    for xi, h in zip(x, rows):
        ax.text(xi + w/2, h[2] + 8, f"{h[1]/h[2]:.1f}x", ha="center", fontsize=7.5, color=S.INK_SOFT)
    ax.set_xticks(x); ax.set_xticklabels([h[0] for h in rows], fontsize=7.5)
    ax.set_ylabel("flash-attention op, ms (pp2048)")
    ax.set_title("The kernel: 3.4 to 4.4 times faster", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False); ax.grid(axis="x", visible=False)

    ax = axes[1]
    x = np.arange(len(HEADS))
    gain = [(h[4] / h[3] - 1) * 100 for h in HEADS]
    ax.bar(x, gain, 0.55, color=S.ROCM)
    for xi, g, h in zip(x, gain, HEADS):
        ax.text(xi, g + 1, f"+{g:.0f} %\n{h[3]:.0f} to {h[4]:.0f}", ha="center", fontsize=7, color=S.INK_SOFT)
    ax.set_xticks(x); ax.set_xticklabels([h[0] for h in HEADS], fontsize=6.8)
    ax.set_ylabel("pp2048 gain, percent, -fa on")
    ax.set_ylim(0, 54)
    ax.set_title("End to end: as large as attention's share", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.06,
             "Same board, builds interleaved, final rows. The 14B models have 40 heads over 8 KV heads and take the 32-column tile\n"
             "without GQA sharing; the MoE and the 27B spend their prefill in matmuls, so a 3.5x faster kernel moves them 6 to 10 percent.",
             fontsize=7, color=S.NOTE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-fa-heads.png"), bbox_inches="tight")
    print("fig-fa-heads.png")


def fig_spill():
    ncol = 2 if SPILL_DEPTH else 1
    fig, axes = plt.subplots(1, ncol, figsize=(8.6 if ncol == 2 else 4.6, 3.4))
    ax = axes[0] if ncol == 2 else axes
    x = np.arange(len(SPILL))
    vg = [s[1] for s in SPILL]; sp = [s[2] for s in SPILL]
    ax.bar(x, vg, 0.62, color=[S.AMBER if s[2] else S.TEAL for s in SPILL], label="VGPRs used")
    ax.bar(x, sp, 0.62, bottom=vg, color=S.AMBER_PALE, edgecolor=S.AMBER, linewidth=0.6,
           hatch="///", label="VGPRs spilled")
    ax.axhline(256, color=S.NOTE, linewidth=1, linestyle="--")
    ax.text(len(SPILL) - 0.45, 268, "256 VGPR budget", fontsize=7, ha="right", color=S.NOTE)
    for xi, s in zip(x, SPILL):
        if s[2]:
            ax.text(xi, s[1] + s[2] + 24, f"+{s[2]}\n{s[3]} B/lane", ha="center",
                    fontsize=7, color=S.INK_SOFT)
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in SPILL], fontsize=7.5)
    ax.set_ylabel("registers per lane")
    ax.set_ylim(0, 1000)
    ax.set_title("Only RDNA1 overruns the register budget", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)

    if SPILL_DEPTH:
        ax2 = axes[1]
        ds = sorted(SPILL_DEPTH)
        base_on = [SPILL_DEPTH[d][1] for d in ds]
        fix_on = [SPILL_DEPTH[d][2] for d in ds]
        off = [SPILL_DEPTH[d][0] for d in ds]
        ax2.plot(ds, fix_on, "-o", color=S.ROCM, ms=5, label="patched, -fa on")
        ax2.plot(ds, off, "-^", color=S.GREY_MID, ms=4.5, label="-fa off (either build)")
        ax2.plot(ds, base_on, "--s", color=S.ROCM_MID, ms=5, label="unpatched, -fa on")
        for d, v in zip(ds, fix_on):
            ax2.text(d, v + 9, f"{v:.0f}", fontsize=7, ha="center", color=S.INK_SOFT)
        for d, v in zip(ds, base_on):
            ax2.text(d, v - 17, f"{v:.0f}", fontsize=7, ha="center", color=S.NOTE)
        ax2.set_xlabel("tokens of context already present")
        ax2.set_ylabel("prefill tokens/s (pp2048)")
        ax2.set_xticks(ds); ax2.set_ylim(0, max(fix_on) * 1.25)
        ax2.set_title("qwen3-8B prefill: the trade disappears", fontsize=9)
        ax2.legend(fontsize=7.5, frameon=False, loc="lower left")

    caption = ("flash_attn_tile<128,128> resource usage from hipcc. RDNA1 has no v_dot2_f32_f16, so ggml_cuda_mad unpacks\n"
               "each half2 into two floats and the kernel spills to scratch on every inner iteration.")
    if SPILL_DEPTH:
        caption = "Left: " + caption + " Right: what removing it buys."
    fig.text(0.01, -0.05, caption, fontsize=7, color=S.NOTE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-rdna1-fattn-spill.png"), bbox_inches="tight")
    print("fig-rdna1-fattn-spill.png")



# ============ decode: MMVQ table and v_sad_u8 (logs/rdna1-mmvq-2026-09-18) ============
# tg128, -fa on, medians of two passes; label -> (four patches, + RDNA2 table, + v_sad_u8, Vulkan)
MMVQ = [  # label -> (four patches, + RDNA2 table, + v_sad_u8 + type-aware table, + float q4_K, + float q6_K/q8_0, + q5_K/IQ/experts, Vulkan); None = not in that A/B
    ("qwen2.5-1.5B\nQ4_K_M",   112.8, 142.4, 146.0, 168.4, 181.4, 180.8, 212.4),
    ("qwen3-8B\nQ8_0",         34.8,  36.3,  36.6,  36.6,  36.7,  None,  39.1),
    ("qwen3-14B\nQ4_K_M",      20.4,  23.4,  24.0,  28.8,  30.4,  None,  34.8),
    ("qwen3.6-35B MoE\nIQ2_M", 33.4,  46.6,  52.1,  52.1,  55.4,  65.9,  87.0),
    ("qwen3.8-27B\nIQ3_XXS",   None,  None,  None,  None,  11.1,  13.9,  17.6),
]

def fig_mmvq():
    fig, ax = plt.subplots(figsize=(9.6, 3.7))
    x = np.arange(len(MMVQ)); w = 0.12
    cols = [("four patches", S.REDS[0], 1), ("+ RDNA2 matvec table", S.REDS[1], 2),
            ("+ v_sad_u8, type-aware table", S.REDS[2], 3), ("+ float q4_K matvec", S.REDS[3], 4),
            ("+ float q6_K, q8_0", S.REDS[4], 5), ("+ q5_K, IQ types, experts (patch 5)", S.REDS[5], 6)]
    for k, (name, col, idx) in enumerate(cols):
        vals = [m[idx] if m[idx] is not None else 0 for m in MMVQ]
        ax.bar(x + (k - 2.5) * w, vals, w, color=col, label=name)
        for xi, m in zip(x, MMVQ):
            if m[idx] is not None:
                ax.text(xi + (k - 2.5) * w, m[idx] + 2, f"{m[idx]:.0f}", ha="center", fontsize=5.8, color=S.INK_SOFT)
    ax.scatter(x + 3.6 * w, [m[7] for m in MMVQ], marker="_", s=260, color=S.VULKAN, linewidths=2.4, label="Vulkan", zorder=3)
    for xi, m in zip(x, MMVQ):
        ax.text(xi + 3.6 * w, m[7] + 2, f"{m[7]:.0f}", ha="center", fontsize=6.8, color=S.VULKAN)
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in MMVQ], fontsize=8)
    ax.set_ylabel("decode tokens/s (tg128)")
    ax.set_ylim(0, 235)
    ax.set_title("Decode: launch geometry, activation sums, then float activations type by type", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.text(0.01, -0.06,
             "RDNA1 was not in the matrix-vector parameter table and ran the generic entry, four warps per row with a barrier. The last three\n"
             "steps replace the emulated int8 dot with float activations: q4_K, then q6_K and q8_0, then q5_K and the IQ2/IQ3 types with the MoE's\n"
             "experts. A bar is missing where a model was not in that A/B. Gates bit-identical through the first three; the float kernels match the CPU.",
             fontsize=7, color=S.NOTE)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig-mmvq-decode.png"), bbox_inches="tight")
    print("fig-mmvq-decode.png")

if __name__ == "__main__":
    fig_backends(); fig_depth(); fig_sgemm(); fig_gemm_shapes()
    fig_op_perf(); fig_spill(); fig_heads(); fig_mmvq()
    fig_f44_backends(); fig_f43_vs_f44(); fig_probes(); fig_f44_depth()
    fig_fa_tradeoff(); fig_prefill_depth(); fig_prefill_depth_24k(); fig_decode_depth_8b(); fig_decode_depth_six(); fig_soak()
