#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""What a decode token is made of, and what the part outside the kernels scales with.

Driven by the numbers in logs/decode-residue-2026-09-25/README.md, which are in turn read off the
kerntrace tables and llama-bench runs in that directory.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)

DARK, MID, ACC = "#2b2b2b", "#9a9a9a", "#555555"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 150,
})

# model, tokens a second untraced, kernel ms a token, dispatches a token
M = [("qwen2.5-1.5B", 183.82, 4.783, 369),
     ("qwen3.6-35B MoE", 67.45, 12.629, 1298)]

names = [m[0] for m in M]
token = [1000.0 / m[1] for m in M]
kern = [m[2] for m in M]
disp = [m[3] for m in M]
resid = [t - k for t, k in zip(token, kern)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 3.2))

x = np.arange(len(M))
a1.bar(x, kern, 0.55, color=DARK, label="inside kernels")
a1.bar(x, resid, 0.55, bottom=kern, color=MID, label="outside any kernel")
for i, (k, r) in enumerate(zip(kern, resid)):
    a1.annotate(f"{r:.2f} ms\n{100*r/(k+r):.0f} %", (i, k + r), ha="center", va="bottom", fontsize=7.5)
a1.set_xticks(x, names, fontsize=8)
a1.set_ylabel("milliseconds a decoded token")
a1.set_ylim(0, max(token) * 1.28)
a1.legend(frameon=False, fontsize=8, loc="upper left")
a1.set_title("a decode token, untraced", fontsize=9)

a2.plot([0, max(disp) * 1.1], [0, max(disp) * 1.1 * 1.78 / 1000], color=ACC, ls="--", lw=0.9,
        label="1.78 us a dispatch")
a2.plot(disp, resid, "o", color=DARK, ms=6)
for i, (d, r, n) in enumerate(zip(disp, resid, names)):
    # the left point sits against the axis, so its label goes to the right of it
    dx, ha = ((10, "left") if i == 0 else (-8, "right"))
    a2.annotate(f"{n}\n{1000*r/d:.2f} us a dispatch", (d, r), textcoords="offset points",
                xytext=(dx, 6), fontsize=7.5, ha=ha)
a2.set_xlim(0, max(disp) * 1.1)
a2.set_ylim(0, max(resid) * 1.35)
a2.set_xlabel("kernel dispatches a token")
a2.set_ylabel("time outside any kernel, ms a token")
a2.legend(frameon=False, fontsize=8, loc="lower right")
a2.set_title("and what the outside part tracks", fontsize=9)

fig.suptitle("gfx1013: the part of a decode token that is not kernel time is per-dispatch cost",
             fontsize=9.5)
fig.tight_layout()
p = os.path.join(OUT, "decode-residue.png")
fig.savefig(p, bbox_inches="tight")
print("wrote", p)
