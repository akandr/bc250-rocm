#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Compare two test-backend-ops perf logs op by op."""
import re
import sys


def load(path):
    out = {}
    name = None
    for line in open(path):
        line = line.rstrip()
        m = re.match(r"\s{2}([A-Z_0-9]+\(name=.*?\)):", line)
        if m:
            name = m.group(1)
        t = re.search(r"([\d.]+) us/run", line)
        if t and name:
            out[name] = float(t.group(1))
            name = None
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
common = [k for k in a if k in b]
rows = sorted(common, key=lambda k: -(a[k] - b[k]))
want = sys.argv[3] if len(sys.argv) > 3 else ""
print(f"{'us hip':>9s} {'us vk':>9s} {'ratio':>6s} {'hip-vk us':>10s}  op")
tot_a = tot_b = 0.0
for k in rows:
    if want and want not in k:
        continue
    tot_a += a[k]
    tot_b += b[k]
for k in rows[:28]:
    if want and want not in k:
        continue
    short = re.sub(r"^([A-Z_0-9]+)\(name=([^,]+),.*?ne=(\[[^\]]*\]).*", r"\1 \2 \3", k)
    print(f"{a[k]:9.2f} {b[k]:9.2f} {a[k]/b[k]:6.2f} {a[k]-b[k]:10.2f}  {short[:96]}")
print(f"\n{len(common)} shared ops; summed {tot_a:.0f} us hip vs {tot_b:.0f} us vulkan (sum is not a token: "
      f"each op carries per-op submission overhead and unnamed nodes lose their layer weight)")
