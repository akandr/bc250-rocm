#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Figures from the gfx1013 hardware counter runs, repo style, grayscale.

Driven straight from the rocprofv3 counter_collection.csv files so the plots and the text cannot
drift apart, which is the failure this repository has already had once with transcribed literals.

    python3 make_counter_figures.py logs/hw-counters-2026-09-25
"""
import csv
import collections
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

DARK = "#2b2b2b"
MID = "#9a9a9a"
ACC = "#555555"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 150,
})


def read_dispatches(path):
    """One dict per dispatch: counters plus kernel name, grid, and device duration in ns."""
    per = collections.OrderedDict()
    with open(path) as f:
        for r in csv.DictReader(f):
            d = int(r["Dispatch_Id"])
            e = per.setdefault(d, {
                "kernel": r["Kernel_Name"],
                "grid": int(r["Grid_Size"]),
                "wg": int(r["Workgroup_Size"]),
                "ns": int(r["End_Timestamp"]) - int(r["Start_Timestamp"]),
                "counters": {},
            })
            e["counters"][r["Counter_Name"]] = float(r["Counter_Value"])
    return list(per.values())


def load(logdir, tag):
    """Every dispatch of one counter run, or an empty list if that run is not present."""
    path = os.path.join(logdir, tag, "run_counter_collection.csv")
    return read_dispatches(path) if os.path.exists(path) else []


def fig_validation(logdir):
    """Predicted against measured instruction counts, the gate the rest of the figures rest on.

    gfx1013 is absent from rocprofiler-sdk's counter_defs.yaml; the counters here come from adding it
    to the gfx1010 lists, which is an assumption about the hardware block layout. This figure is the
    evidence that the assumption holds: a kernel whose VALU count is fixed by its ISA, run at four
    sizes, reporting the predicted number every time.
    """
    sizes, measured, predicted = [], [], []
    for blocks in (64, 256, 1024, 2560):
        path = os.path.join(logdir, f"validate_{blocks}", "run_counter_collection.csv")
        if not os.path.exists(path):
            continue
        d = read_dispatches(path)[0]
        sizes.append(int(d["counters"]["SQ_WAVES"]))
        measured.append(d["counters"]["SQ_INSTS_VALU"])
        predicted.append(int(d["counters"]["SQ_WAVES"]) * 8192)
    if not sizes:
        return None

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    x = np.arange(len(sizes))
    w = 0.38
    ax.bar(x - w / 2, np.array(predicted) / 1e6, w, color=DARK, label="predicted from the ISA")
    ax.bar(x + w / 2, np.array(measured) / 1e6, w, color=MID, label="SQ_INSTS_VALU")
    ax.set_xticks(x, [f"{s}" for s in sizes])
    ax.set_xlabel("waves launched")
    ax.set_ylabel("VALU instructions, millions")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("counts", fontsize=9)

    per = np.array(measured) / np.array(sizes)
    ax2.plot(x, per, "o-", color=DARK, ms=4)
    ax2.axhline(8192, color=ACC, ls="--", lw=0.8)
    ax2.annotate("8192, the FMAs in the loop", (0, 8192), textcoords="offset points",
                 xytext=(4, -12), fontsize=7.5, color=ACC)
    ax2.set_xticks(x, [f"{s}" for s in sizes])
    ax2.set_xlabel("waves launched")
    ax2.set_ylabel("VALU instructions per wave")
    ax2.set_ylim(8100, 8300)
    ax2.set_title("per wave, constant to the instruction", fontsize=9)

    fig.suptitle("gfx1013 counters against a kernel with a known instruction count", fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(OUT, "counter-validation.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# Flops one instruction retires across a wave32: two per FMA, doubled again when the FMA is packed,
# times the 32 lanes.
FLOPS_PER_INSTR = {"fp32": 2 * 32, "fp16": 4 * 32, "fp64": 2 * 32}
CEILING = {"fp32": 6.52, "fp16": 13.02, "fp64": 0.43}

# Rates measured on this board with the profiler detached, square 8192 except DGEMM at 4096.
# logs/hw-counters-2026-09-25/plain_rocblas2.txt and plain_pk_bk32.txt.
RATES = {
    "rocBLAS DGEMM": 0.457,
    "rocBLAS SGEMM": 4.593,
    "rocBLAS HGEMM": 4.648,
    "hand-written, BK 32": 8.90,
    "hand-written, BK 16": 2.31,
}


def largest_per_kernel(ds):
    best = {}
    for d in ds:
        if d["grid"] >= best.get(d["kernel"], {"grid": 0})["grid"]:
            best[d["kernel"]] = d
    return best


def one(logdir, tag, want=None):
    """The largest dispatch in a run, optionally restricted to kernels whose name contains `want`."""
    ds = [d for d in load(logdir, tag) if want is None or want in d["kernel"]]
    ds = [d for d in ds if d["grid"] > 200000]
    return max(ds, key=lambda d: d["grid"]) if ds else None


def collect(logdir):
    """The four GEMMs at their largest square, with instruction counts and residency."""
    out = []
    for want, prec, name, n in (("_DB_", "fp64", "rocBLAS DGEMM", 4096),
                                ("_SB_", "fp32", "rocBLAS SGEMM", 8192),
                                ("_HB_", "fp16", "rocBLAS HGEMM", 8192)):
        i, w = one(logdir, "i_rocblas", want), one(logdir, "w_rocblas", want)
        if i:
            out.append({"name": name, "prec": prec, "n": n, "i": i, "w": w})
    for tag, wtag, name in (("pk_bk32", "w_bk32", "hand-written, BK 32"),
                            ("pk_bk16", "w_bk16", "hand-written, BK 16")):
        i, w = one(logdir, tag), one(logdir, wtag)
        if i:
            out.append({"name": name, "prec": "fp16", "n": 8192, "i": i, "w": w})
    return out


def fig_gemm(logdir):
    """The three quantities that separate these kernels, beside the speed each reaches.

    Instruction counts come from profiled runs, where they are exact; the rates come from runs with
    the profiler detached, because counter collection costs these kernels very unequal amounts.
    """
    rows = collect(logdir)
    if not rows:
        return None
    short = {"rocBLAS DGEMM": "DGEMM\nfp64", "rocBLAS SGEMM": "SGEMM\nfp32",
             "rocBLAS HGEMM": "HGEMM\nfp16", "hand-written, BK 32": "ours\nBK 32",
             "hand-written, BK 16": "ours\nBK 16"}
    labels = [short.get(r["name"], r["name"]) for r in rows]
    x = np.arange(len(rows))
    colors = [DARK if "hand-written" in r["name"] else MID for r in rows]

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.3))

    valu = [r["i"]["counters"]["SQ_INSTS_VALU"] / 1e9 for r in rows]
    mins = [2.0 * r["n"] ** 3 / FLOPS_PER_INSTR[r["prec"]] / 1e9 for r in rows]
    axes[0].bar(x, valu, 0.62, color=colors)
    for i, m in enumerate(mins):
        axes[0].plot([i - 0.34, i + 0.34], [m, m], color=ACC, lw=1.2)
    axes[0].set_ylabel("VALU instructions, billions")
    axes[0].set_title("instructions issued\n(bar) against the fewest\nthe precision allows (line)",
                      fontsize=8.5)

    ratio = [r["i"]["counters"]["SQ_INSTS_VALU"] / r["i"]["counters"]["SQ_INSTS_LDS"] for r in rows]
    axes[1].bar(x, ratio, 0.62, color=colors)
    for i, v in enumerate(ratio):
        axes[1].annotate(f"{v:.1f}", (i, v), ha="center", va="bottom", fontsize=7.5)
    axes[1].set_ylabel("VALU instructions per LDS instruction")
    axes[1].set_title("arithmetic done per\ntrip to shared memory", fontsize=8.5)

    res = [r["w"]["counters"]["SQ_WAVE_CYCLES"] / r["w"]["counters"]["SQ_BUSY_CYCLES"]
           if r["w"] else float("nan") for r in rows]
    axes[2].bar(x, res, 0.62, color=colors)
    axes[2].set_ylabel("wave-cycles per busy cycle")
    axes[2].set_title("waves kept resident", fontsize=8.5)

    tf = [RATES.get(r["name"], float("nan")) for r in rows]
    axes[3].bar(x, tf, 0.62, color=colors)
    for i, r in enumerate(rows):
        c = CEILING[r["prec"]]
        axes[3].plot([i - 0.34, i + 0.34], [c, c], color=ACC, lw=1.2)
        if c < 1.0:      # fp64's 0.43 sits on the axis and cannot be told from the bar
            axes[3].annotate(f"ceiling {c}", (i, c), textcoords="offset points", xytext=(0, 6),
                             fontsize=6.8, color=ACC, ha="center")
    axes[3].set_ylabel("TFLOP/s, profiler detached")
    axes[3].set_title("speed reached (bar)\nagainst the measured\nALU ceiling (line)", fontsize=8.5)

    for a in axes:
        a.set_xticks(x, labels, fontsize=7.8)
    fig.suptitle("gfx1013, square GEMM at N=8192, DGEMM at 4096. The first three are rocBLAS, the "
                 "last two are this repository's packed-fp16 kernel at two tile depths.\n"
                 "What separates them is not how many instructions they issue.", fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(OUT, "counter-gemm.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_scalarised(logdir):
    """The int8 row of the ALU sweep, before and after the accumulators were made lane-dependent.

    Seeded from the kernel argument the whole int8 dot product is uniform across the wave and the
    compiler moves it to the scalar unit. Floating point cannot go that way, since RDNA1 has no
    scalar float ALU, so only the integer row was affected: the control kernels are unchanged.
    """
    before, after = load(logdir, "alu_orig_c"), load(logdir, "alu_fixed_c")
    if not before or not after:
        return None

    def per_wave(ds, kernel):
        for d in ds:
            if d["kernel"].startswith(kernel) and d["counters"].get("SQ_WAVES") == 640:
                w = d["counters"]["SQ_WAVES"]
                return d["counters"]["SQ_INSTS_VALU"] / w, d["counters"]["SQ_INSTS_SALU"] / w
        return None

    names = [("c_dp4a", "int8 dot product\n(emulated)"), ("c_f32", "v_fma_f32\n(control)")]
    got = [(lbl, per_wave(before, k), per_wave(after, k)) for k, lbl in names]
    got = [g for g in got if g[1] and g[2]]
    if not got:
        return None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    x = np.arange(len(got))
    w = 0.36
    for ax, idx, title in ((a1, 0, "vector instructions per wave"),
                           (a2, 1, "scalar instructions per wave")):
        ax.bar(x - w / 2, [g[1][idx] for g in got], w, color=MID, label="seeded from the argument")
        ax.bar(x + w / 2, [g[2][idx] for g in got], w, color=DARK, label="seeded from the lane index")
        ax.set_yscale("log")
        ax.set_xticks(x, [g[0] for g in got], fontsize=7.5)
        ax.set_title(title, fontsize=9)
    a1.set_ylabel("instructions per wave, log scale")
    a2.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle("gfx1013: an int8 benchmark that was measuring the scalar unit", fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(OUT, "counter-scalarised.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    logdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "logs", "hw-counters-2026-09-25")
    for f in (fig_validation(logdir), fig_gemm(logdir), fig_scalarised(logdir)):
        if f:
            print("wrote", f)
