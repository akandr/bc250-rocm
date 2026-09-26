#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Split a decode token into kernel time, copy time, and the gaps between them.

logs/moe-kernels-reweighted-2026-09-23/ left 1.81 ms of the MoE's 13.93 ms token unaccounted for:
kerntrace measured 12.12 ms inside kernels and nothing measured what the rest was. That needed the
HIP API timeline beside the dispatch timeline on one run, which needed a profiler this board did
not have until logs/hw-counters-2026-09-25/.

Reads the CSVs rocprofv3 writes with --kernel-trace --memory-copy-trace --hip-runtime-trace and
reports, over a window of steady decode:

  busy      GPU occupied by a kernel or a copy
  gap       no kernel and no copy in flight
  covered   of that gap, the part where a HIP API call was open on the host, so the host was
            inside the runtime instead of waiting on it
  uncovered the rest, where neither the GPU nor the HIP runtime was doing anything visible

Tracing inflates the gaps, which is exactly the quantity of interest, so the absolute gap figure
from a traced run is an upper bound and is reported as one. What survives is the split between
covered and uncovered, and the per-dispatch cost, both of which are ratios within the same run.

    python3 decode_timeline.py <rocprofv3 output dir> [--tokens N]
"""
import csv
import os
import sys


def read(path, cols):
    """Rows of one rocprofv3 trace CSV, as dicts, or an empty list if it is not there."""
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if all(c in r for c in cols):
                out.append(r)
    return out


def spans(rows, lo="Start_Timestamp", hi="End_Timestamp"):
    return [(int(r[lo]), int(r[hi])) for r in rows if r.get(lo) and r.get(hi)]


def merge(intervals):
    """Union of half-open intervals, as a sorted disjoint list."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def total(intervals):
    return sum(b - a for a, b in intervals)


def overlap(a, b):
    """Total length of the intersection of two disjoint sorted interval lists."""
    i = j = 0
    acc = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            acc += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return acc


def complement(window, busy):
    """The parts of `window` not covered by `busy`."""
    lo, hi = window
    out = []
    cur = lo
    for a, b in busy:
        if b <= lo or a >= hi:
            continue
        a, b = max(a, lo), min(b, hi)
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        out.append((cur, hi))
    return out


def find_period(rows, maxp=4000):
    """Dispatches per decoded token, found exactly, not by looking for a regular kernel.

    Every token runs the same decode graph, so the sequence of kernel names repeats with a period
    equal to the dispatches in one token. Looking instead for a kernel that fires once per token is
    unreliable: most fire once per layer, and the few that do not are too few to judge regularity
    from. This takes the smallest period p for which the last p names repeat the p before them.
    """
    names = [r["Kernel_Name"] for r in rows]
    n = len(names)
    for p in range(8, min(maxp, n // 3)):
        if names[n - p:] == names[n - 2 * p:n - p]:
            return p
    return None


def main(outdir, want_tokens=None):
    kern = read(os.path.join(outdir, "run_kernel_trace.csv"), ["Start_Timestamp"])
    copy = read(os.path.join(outdir, "run_memory_copy_trace.csv"), ["Start_Timestamp"])
    api = read(os.path.join(outdir, "run_hip_api_trace.csv"), ["Start_Timestamp"])
    print(f"{len(kern):,} kernel dispatches, {len(copy):,} copies, {len(api):,} HIP API calls\n")
    if not kern:
        print("no kernel trace found in", outdir)
        return

    kern.sort(key=lambda r: int(r["Start_Timestamp"]))
    # The runtime's own helpers (buffer fills and copies) fire on their own schedule, so the decode
    # graph's period is only visible in the model's kernels. They still count as GPU-busy below.
    model = [r for r in kern if not r["Kernel_Name"].startswith("__amd_rocclr_")]
    period = find_period(model)
    if not period:
        print("the dispatch sequence does not repeat; cannot cut the timeline into tokens")
        return
    print(f"decode graph repeats every {period} dispatches\n")

    tokens = want_tokens or 8
    have = len(model) // period
    tokens = min(tokens, have - 1)      # drop one token's worth as a guard against the tail
    if tokens < 2:
        print(f"only {have} whole graphs in the trace, too few")
        return
    first = model[len(model) - tokens * period]
    window = (int(first["Start_Timestamp"]), int(model[-1]["End_Timestamp"]))
    span = window[1] - window[0]

    kb = merge([s for s in spans(kern) if s[1] > window[0] and s[0] < window[1]])
    cb = merge([s for s in spans(copy) if s[1] > window[0] and s[0] < window[1]])
    busy = merge(kb + cb)
    gaps = complement(window, busy)
    ab = merge([s for s in spans(api) if s[1] > window[0] and s[0] < window[1]])
    covered = overlap(gaps, ab)

    ndisp = sum(1 for s in spans(kern) if window[0] <= s[0] < window[1])
    ncopy = sum(1 for s in spans(copy) if window[0] <= s[0] < window[1])

    def per(x):
        return x / tokens / 1e6

    print(f"window {span/1e6:.1f} ms over {tokens} tokens, "
          f"{ndisp/tokens:.0f} dispatches and {ncopy/tokens:.1f} copies a token\n")
    print(f"{'':<34}{'ms a token':>12}{'share':>9}")
    print(f"{'token, traced':<34}{per(span):>12.3f}{100.0:>8.1f}%")
    print(f"{'  kernels':<34}{per(total(kb)):>12.3f}{100*total(kb)/span:>8.1f}%")
    print(f"{'  copies, not overlapping a kernel':<34}"
          f"{per(total(busy)-total(kb)):>12.3f}{100*(total(busy)-total(kb))/span:>8.1f}%")
    print(f"{'  gap, no kernel and no copy':<34}{per(total(gaps)):>12.3f}{100*total(gaps)/span:>8.1f}%")
    print(f"{'    host inside a HIP call':<34}{per(covered):>12.3f}{100*covered/span:>8.1f}%")
    print(f"{'    neither':<34}{per(total(gaps)-covered):>12.3f}"
          f"{100*(total(gaps)-covered)/span:>8.1f}%")
    if ndisp:
        print(f"\ngap per dispatch {total(gaps)/ndisp/1e3:.2f} us, "
              f"of which {covered/ndisp/1e3:.2f} us inside a HIP call")
    print("\nThe gap row is tracing overhead almost in its entirety and must not be read as the\n"
          "kernels' real spacing. On the 1.5B this run measures 5.3 ms of kernel time in a token\n"
          "that takes 5.4 ms untraced, so the true non-kernel time is about a tenth of a\n"
          "millisecond, while the gap here reads 14 ms. Kernel and copy time are what this\n"
          "instrument measures; the gap is what it costs.")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    t = next((int(x.split("=")[1]) for x in sys.argv[1:] if x.startswith("--tokens=")), None)
    main(a[0] if a else ".", t)
