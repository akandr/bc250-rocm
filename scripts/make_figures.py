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
# ROCm values: -fa off -ub 8 (the text-verified configuration). Vulkan: -fa on.
MODELS = {
    "qwen2.5 1.5B Q4_K_M":        (106.80, 210.70, 182.17, 1844.23),
    "deepseek-r1 14B Q4_K_M *":   (19.6, 34.50, None, 198.88),
    "qwen3.6 35B-A3B MoE IQ2_M":  (31.6, 85.59, None, 457.90),
}

# depth ladder (qwen2.5-1.5b): depth -> (hip_tg64, vk_tg64)
DEPTH = {0: (106.80, 210.70), 4096: (74.15, None), 8192: (54.60, 158.00), 16384: (35.35, 137.08), 24576: (None, 126.49), 30720: (None, 116.79)}

# sgemm: N -> median ms/iter (20 iters, all correct)
SGEMM = {512: 0.2, 1024: 0.8, 2048: 5.7, 4096: 30.0, 8192: 236.0}

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
        (a1, hip_tg, vk_tg, "decode (tg128, tokens/s)", False),
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
    a1.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.text(0.01, -0.02, "ROCm at -fa off -ub 8, the configuration whose output passes a text "
             "check. * 14B ROCm decode is tg32 (three loads, 19.5 to 19.7); its ROCm pp512 was "
             "not measured.", fontsize=7, color="#444")
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

if __name__ == "__main__":
    fig_backends(); fig_depth(); fig_sgemm()
