#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Medians from a campaign_split_f44.sh log: one line per model, round and backend, each with the
# three llama-bench samples of pp512 and tg64. Prints a README table and the dict make_figures.py
# uses for the Fedora 44 backend figure.
# Usage: campaign_medians.py <log> [<second log to compare against>]
import re, sys, statistics, collections
pat = re.compile(r"\] (\S+) r(\d) (hip|vk) pp512=\[([^\]]*)\] tg64=\[([^\]]*)\]")

def load(path):
    out = collections.defaultdict(list)   # (model, be, metric) -> samples
    for line in open(path):
        m = pat.search(line)
        if not m: continue
        model, rnd, be, pp, tg = m.groups()
        for metric, s in (("pp", pp), ("tg", tg)):
            out[(model, be, metric)] += [float(x) for x in s.split(",") if x.strip()]
    return out

med = lambda xs: statistics.median(xs) if xs else float("nan")
a = load(sys.argv[1]); b = load(sys.argv[2]) if len(sys.argv) > 2 else None
models = []
for (m, _, _) in a:
    if m not in models: models.append(m)

print("| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |")
print("|---|---|---|---|---|")
for m in models:
    r = [med(a[(m, be, k)]) for be in ("hip", "vk") for k in ("pp", "tg")]
    print(f"| {m} | {r[0]:.1f} | {r[2]:.1f} | {r[1]:.1f} | {r[3]:.1f} |")

print("\nspread (max-min over the nine samples, percent of median):")
for m in models:
    s = []
    for be in ("hip", "vk"):
        for k in ("pp", "tg"):
            xs = a[(m, be, k)]
            s.append(f"{be} {k} {((max(xs)-min(xs))/med(xs)*100 if xs else float('nan')):.1f}")
    print(f"  {m:26s} " + "  ".join(s))

if b:
    print("\nagainst the second log (this / other, medians):")
    for m in models:
        row = []
        for be in ("hip", "vk"):
            for k in ("pp", "tg"):
                x, y = med(a[(m, be, k)]), med(b[(m, be, k)])
                row.append(f"{be} {k} {x/y:.3f}" if y == y else f"{be} {k} n/a")
        print(f"  {m:26s} " + "  ".join(row))

print("\nF44 = {  # model: (hip tg64, vk tg64, hip pp512, vk pp512)")
for m in models:
    print(f'    "{m}": ({med(a[(m,"hip","tg")]):.2f}, {med(a[(m,"vk","tg")]):.2f}, {med(a[(m,"hip","pp")]):.2f}, {med(a[(m,"vk","pp")]):.2f}),')
print("}")
