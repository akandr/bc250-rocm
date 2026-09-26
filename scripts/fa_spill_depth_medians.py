#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Turn scripts/fa_spill_depth.sh's log into per-point medians and the SPILL_DEPTH dict for make_figures.py.
# Usage: fa_spill_depth_medians.py logs/rdna1-fattn-spill-2026-09-17/depth-sweep.log
import re, sys, statistics, collections
pat = re.compile(r"pass (\d+) d=(\d+) fa=(\d) (base|fixed) ([\d.]+)")
vals = collections.defaultdict(list)   # (d, fa, arm) -> [values]
for line in open(sys.argv[1]):
    m = pat.search(line)
    if m:
        p, d, fa, arm, v = m.groups()
        vals[(int(d), int(fa), arm)].append((int(p), float(v)))

depths = sorted({k[0] for k in vals})
print(f"{'depth':>6} {'arm':>6} {'fa':>3}  passes -> median")
for d in depths:
    for arm in ("base", "fixed"):
        for fa in (0, 1):
            xs = vals.get((d, fa, arm), [])
            if not xs: continue
            med = statistics.median(v for _, v in xs)
            spread = (max(v for _, v in xs) - min(v for _, v in xs)) / med * 100 if len(xs) > 1 else 0
            print(f"{d:>6} {arm:>6} {fa:>3}  {' '.join(f'{v:7.2f}' for _, v in xs):>26} -> {med:7.2f}  (spread {spread:.1f} %)")

# control check: the -fa off arms run identical code, so they should agree per pass
print("\ncontrol: -fa off base vs fixed per pass (should agree)")
for d in depths:
    b = dict(vals.get((d, 0, "base"), [])); f = dict(vals.get((d, 0, "fixed"), []))
    for p in sorted(set(b) & set(f)):
        diff = (f[p] - b[p]) / b[p] * 100
        flag = "  <-- contaminated pass" if abs(diff) > 4 else ""
        print(f"  d={d:<5} pass {p}: base {b[p]:7.2f} fixed {f[p]:7.2f}  {diff:+.1f} %{flag}")

print("\nSPILL_DEPTH = {")
for d in depths:
    row = []
    for arm, fa in (("base", 0), ("base", 1), ("fixed", 0), ("fixed", 1)):
        xs = vals.get((d, fa, arm), [])
        row.append(statistics.median(v for _, v in xs) if xs else float("nan"))
    print(f"    {d}: ({row[0]:.2f}, {row[1]:.2f}, {row[2]:.2f}, {row[3]:.2f}),")
print("}")
