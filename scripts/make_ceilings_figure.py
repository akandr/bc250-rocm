#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""How deep each model can still generate, ROCm against Vulkan, by KV cache type.

Read from logs/context-ceilings-2026-09-30/ceilings.tsv and trained-context.txt, so nothing here is
typed in by hand. ROCm is the hip arm, as shipped. Each bar is the deepest depth that generated; the
f16 part is the full colour, and the lighter parts are how much further q8_0 and q4_0 went. The tick on each
model's rows is the context length the model was trained on. Colours come from figstyle.py.
"""
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "logs", "context-ceilings-2026-09-30")
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, HERE)
import figstyle as S
S.apply()
OUT = S.out(OUT)
VK, HIP = S.VULKAN, S.ROCM
SHADE = {"hip": {"f16": S.ROCM, "q8_0": S.ROCM_MID, "q4_0": S.ROCM_PALE},
         "vk": {"f16": S.VULKAN, "q8_0": S.VULKAN_MID, "q4_0": S.VULKAN_PALE}}

ceil = {}
with open(os.path.join(LOG, "ceilings.tsv")) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        ceil[(r["arm"], r["model"], r["kv"])] = int(r["generates_at"])
trained = {}
for line in open(os.path.join(LOG, "trained-context.txt")):
    m = re.match(r"(\S+)\.gguf\s+\S+\s+(\d+)", line)
    if m:
        trained[m[1]] = int(m[2])

MODELS = [("qwen3.8-27b-iq3xxs", "qwen3.8-27B UD-IQ3_XXS"), ("qwen3-14b", "qwen3-14B Q4_K_M"),
          ("qwen3-8b-q8_0", "qwen3-8B Q8_0"), ("qwen3.6-35b-a3b-iq2m", "qwen3.6-35B-A3B MoE IQ2_M"),
          ("qwen2.5-1.5b-q4km", "qwen2.5-1.5B Q4_K_M")]
KVS = ["f16", "q8_0", "q4_0"]
BASE = 8192

fig, ax = plt.subplots(figsize=(7.8, 5.2))
yt, yl = [], []
y = 0.0
for m, label in MODELS:
    ax.text(BASE * 1.04, y, label, va="center", fontsize=8.5, fontweight="bold")
    top = y + 0.6
    y += 0.6
    for arm, col, name in (("hip", HIP, "ROCm"), ("vk", VK, "Vulkan")):
        # widest first, so each larger cache is drawn over the extra depth of the next smaller one
        for kv in reversed(KVS):
            c = ceil[(arm, m, kv)]
            ax.barh(y, c - BASE, left=BASE, height=0.55, color=SHADE[arm][kv], edgecolor="none")
        ax.text(ceil[(arm, m, "q4_0")] * 1.05, y, f"{ceil[(arm, m, 'q4_0')]}", va="center", fontsize=7,
                color=S.INK_SOFT, zorder=4, bbox=dict(facecolor=S.BG, edgecolor="none", pad=0.6))
        yt.append(y)
        yl.append(name)
        y += 0.65
    ax.plot([trained[m]] * 2, [top - 0.35, y - 0.3], color=S.INK, lw=1.2, ls=(0, (2, 1.5)), zorder=3)
    y += 0.35

ax.set_xscale("log", base=2)
ticks = [8192, 32768, 131072, 524288, 2097152]
ax.set_xticks(ticks, ["8k", "32k", "128k", "512k", "2M"])
ax.set_xlim(BASE, 2097152 * 1.2)
ax.set_ylim(y - 0.2, -0.5)
ax.set_yticks(yt, yl, fontsize=7.5)
ax.grid(axis="y", visible=False)
ax.set_xlabel("deepest context that still generated, tokens (llama-bench without its state copy, mmap on)")
ax.legend(handles=[Patch(color=S.ROCM, label="ROCm, f16 cache"),
                   Patch(color=S.ROCM_MID, label="ROCm, further with q8_0"),
                   Patch(color=S.ROCM_PALE, label="ROCm, further with q4_0"),
                   Patch(color=S.VULKAN, label="Vulkan, f16 cache"),
                   Patch(color=S.VULKAN_MID, label="Vulkan, further with q8_0"),
                   Patch(color=S.VULKAN_PALE, label="Vulkan, further with q4_0"),
                   Line2D([], [], color=S.INK, lw=1.2, ls=(0, (2, 1.5)), label="trained context length")],
          frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
fig.suptitle("gfx1013 context ceilings, ROCm against Vulkan\n"
             "ROCm stops at KFD's 13.1 GiB limit; Vulkan goes 1.2 to 2.3 times as deep", fontsize=9)
p = os.path.join(OUT, "fig-context-ceilings.png")
fig.savefig(p, bbox_inches="tight")
print("wrote", p)
