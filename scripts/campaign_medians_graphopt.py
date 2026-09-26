#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Medians from one or more campaign_graphopt.sh logs, which carry three arms, not two:
# ROCm as shipped, ROCm with GGML_CUDA_GRAPH_OPT=1, and Vulkan.
#
# scripts/campaign_medians.py cannot read these logs because its pattern matches only hip|vk, and it
# is left alone because several pages quote its output. Usage:
#   campaign_medians_graphopt.py <log> [<log> ...]
# Several logs are pooled, which is how the campaigns in logs/ are reported: each figure is the median
# of all the samples from every run given.
import collections
import re
import statistics
import sys

PAT = re.compile(r"\] (\S+) r(\d) (hip|hipopt|vk) pp512=\[([^\]]*)\] tg64=\[([^\]]*)\]")

NAMES = {
    "qwen2.5-1.5b-q4km":      "qwen2.5-1.5B Q4_K_M",
    "qwen3-8b-q8_0":          "qwen3-8B Q8_0",
    "deepseek-r1-14b":        "deepseek-r1-14B Q4_K_M",
    "qwen3-14b":              "qwen3-14B Q4_K_M",
    "qwen3.6-35b-a3b-iq2m":   "qwen3.6-35B-A3B MoE IQ2_M",
    "qwen3.8-27b-iq3xxs":     "qwen3.8-27B UD-IQ3_XXS",
}


def load(paths):
    out = collections.defaultdict(list)   # (model, arm, metric) -> samples
    order = []
    for path in paths:
        for line in open(path):
            m = PAT.search(line)
            if not m:
                continue
            model, _rnd, arm, pp, tg = m.groups()
            if model not in order:
                order.append(model)
            for metric, s in (("pp", pp), ("tg", tg)):
                out[(model, arm, metric)] += [float(x) for x in s.split(",") if x.strip()]
    return out, order


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    a, models = load(sys.argv[1:])
    n = len(a[(models[0], "hip", "tg")]) if models else 0
    print(f"pooled over {len(sys.argv)-1} run(s), {n} samples a figure\n")

    print("| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | ROCm tg64, GRAPH_OPT | Vulkan tg64 |")
    print("|---|---|---|---|---|---|")
    for m in models:
        pp_h, pp_v = med(a[(m, "hip", "pp")]), med(a[(m, "vk", "pp")])
        tg_h, tg_o, tg_v = med(a[(m, "hip", "tg")]), med(a[(m, "hipopt", "tg")]), med(a[(m, "vk", "tg")])
        print(f"| {NAMES.get(m, m)} | {pp_h:.1f} | {pp_v:.1f} | {tg_h:.1f} | **{tg_o:.1f}** | {tg_v:.1f} |")

    print("\n| model | decode vs Vulkan, as shipped | with GRAPH_OPT | the option is worth | prefill, as shipped | with GRAPH_OPT |")
    print("|---|---|---|---|---|---|")
    for m in models:
        tg_h, tg_o, tg_v = med(a[(m, "hip", "tg")]), med(a[(m, "hipopt", "tg")]), med(a[(m, "vk", "tg")])
        pp_h, pp_o = med(a[(m, "hip", "pp")]), med(a[(m, "hipopt", "pp")])
        print(f"| {NAMES.get(m, m)} | {tg_h/tg_v:.2f} | **{tg_o/tg_v:.2f}** | {tg_o/tg_h:.3f} | {pp_h:.1f} | {pp_o:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
