#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Print the tables in logs/hw-counters-2026-09-25/README.md straight from the CSVs.

Only quantities that survive the profiler are reported. rocprofv3 serializes dispatches to read the
counters, and it costs these kernels very unequal amounts of wall time, so instruction counts and
ratios between counters in the same clock domain are reported and elapsed-time rates are not.

    python3 counter_summary.py logs/hw-counters-2026-09-25
"""
import csv
import collections
import os
import sys


def dispatches(path, want=None, min_grid=0):
    per = collections.OrderedDict()
    with open(path) as f:
        for r in csv.DictReader(f):
            if want and want not in r["Kernel_Name"]:
                continue
            d = int(r["Dispatch_Id"])
            e = per.setdefault(d, {"kernel": r["Kernel_Name"], "grid": int(r["Grid_Size"]),
                                   "vgpr": int(r["VGPR_Count"]), "scratch": int(r["Scratch_Size"]),
                                   "c": {}})
            e["c"][r["Counter_Name"]] = float(r["Counter_Value"])
    return [d for d in per.values() if d["grid"] >= min_grid]


def biggest(logdir, tag, want=None, min_grid=200000):
    p = os.path.join(logdir, tag, "run_counter_collection.csv")
    if not os.path.exists(p):
        return None
    ds = dispatches(p, want, min_grid)
    return max(ds, key=lambda d: d["grid"]) if ds else None


def validation(logdir):
    print("## Validation: 8192 v_fma_f32 per wave, fixed by the ISA\n")
    print(f"{'waves':>8} {'SQ_WAVES':>10} {'SQ_INSTS_VALU':>16} {'per wave':>10}")
    for b in (64, 256, 1024, 2560):
        d = biggest(logdir, f"validate_{b}", min_grid=0)
        if not d:
            continue
        w, v = d["c"]["SQ_WAVES"], d["c"]["SQ_INSTS_VALU"]
        print(f"{b:>8} {w:>10.0f} {v:>16,.0f} {v/w:>10.1f}")


def gemms(logdir):
    # A v_pk_fma_f16 retires 4 flops across 32 lanes, a v_fma_f32 two.
    N = 8192
    rows = [("rocBLAS SGEMM", biggest(logdir, "i_rocblas", "_SB_"), biggest(logdir, "w_rocblas", "_SB_"), 64),
            ("rocBLAS HGEMM", biggest(logdir, "i_rocblas", "_HB_"), biggest(logdir, "w_rocblas", "_HB_"), 128),
            ("this kernel, BK 32", biggest(logdir, "pk_bk32"), biggest(logdir, "w_bk32"), 128),
            ("this kernel, BK 16", biggest(logdir, "pk_bk16"), biggest(logdir, "w_bk16"), 128)]
    print("\n## Square 8192\n")
    print(f"{'kernel':<20} {'VGPR':>5} {'VALU':>16} {'of min':>7} {'LDS':>14} "
          f"{'VALU/LDS':>9} {'resident':>9} {'wait/wave':>12}")
    for name, i, w, fpi in rows:
        if not i:
            continue
        v, l = i["c"]["SQ_INSTS_VALU"], i["c"]["SQ_INSTS_LDS"]
        mini = 2.0 * N ** 3 / fpi
        res = wait = float("nan")
        if w:
            # both counters are cycles in the same domain, so the ratio survives clock and serialization
            res = w["c"]["SQ_WAVE_CYCLES"] / w["c"]["SQ_BUSY_CYCLES"]
            wait = w["c"]["SQ_WAIT_ANY"] / w["c"]["SQ_WAVES"]
        print(f"{name:<20} {i['vgpr']:>5} {v:>16,.0f} {v/mini:>7.2f} {l:>14,.0f} "
              f"{v/l:>9.2f} {res:>9.1f} {wait:>12,.0f}")

    print("\n## LDS bank conflicts, a ratio of two counters\n")
    for tag, name in (("sqc_bk32", "BK 32"), ("pkgemm_sqc", "BK 16")):
        d = biggest(logdir, tag)
        if not d:
            continue
        bc, ia = d["c"].get("SQC_LDS_BANK_CONFLICT"), d["c"].get("SQC_LDS_IDX_ACTIVE")
        if bc is None or not ia:
            continue
        print(f"  {name}: {bc:,.0f} / {ia:,.0f} = {100*bc/ia:.2f} %")


def scalarisation(logdir):
    print("\n## The int8 row of alu_cycles.cpp, at 640 waves\n")
    print(f"{'kernel':<10} {'seeding':<24} {'VALU/wave':>12} {'SALU/wave':>12}")
    for tag, how in (("alu_orig_c", "from the argument"), ("alu_fixed_c", "from the lane index")):
        p = os.path.join(logdir, tag, "run_counter_collection.csv")
        if not os.path.exists(p):
            continue
        seen = set()
        for d in dispatches(p):
            k = d["kernel"].split("(")[0]
            if k not in ("c_dp4a", "c_f32") or k in seen or d["c"].get("SQ_WAVES") != 640:
                continue
            seen.add(k)
            w = d["c"]["SQ_WAVES"]
            print(f"{k:<10} {how:<24} {d['c']['SQ_INSTS_VALU']/w:>12,.0f} "
                  f"{d['c']['SQ_INSTS_SALU']/w:>12,.0f}")


def alu_cost(logdir):
    print("\n## SQ_BUSY_CYCLES per VALU instruction at 640 waves\n")
    p = os.path.join(logdir, "alu", "run_counter_collection.csv")
    if not os.path.exists(p):
        return
    got = collections.defaultdict(list)
    for d in dispatches(p):
        k = d["kernel"].split("(")[0]
        if d["c"].get("SQ_WAVES") == 640 and d["c"].get("SQ_INSTS_VALU"):
            got[k].append(d["c"]["SQ_BUSY_CYCLES"] / d["c"]["SQ_INSTS_VALU"])
    for k in ("c_f32", "c_f16", "c_f16x2"):
        if got[k]:
            v = sorted(got[k])
            print(f"  {k:<10} {v[len(v)//2]:.4f}")


def llama(logdir):
    print("\n## llama.cpp, qwen2.5-1.5B q4_K_M\n")
    for tag in ("llama_decode", "llama_prefill"):
        p = os.path.join(logdir, tag, "run_counter_collection.csv")
        if not os.path.exists(p):
            continue
        per = collections.defaultdict(lambda: collections.Counter())
        total = collections.Counter()
        for r in csv.DictReader(open(p)):
            per[r["Kernel_Name"].split("<")[0][:42]][r["Counter_Name"]] += float(r["Counter_Value"])
            total[r["Counter_Name"]] += float(r["Counter_Value"])
        top = max(per.items(), key=lambda kv: kv[1]["SQ_INSTS_VALU"])
        v, l = top[1]["SQ_INSTS_VALU"], top[1]["SQ_INSTS_LDS"]
        print(f"  {tag:<14} VALU {total['SQ_INSTS_VALU']:>16,.0f}   dominant {top[0]}"
              f"  {100*v/total['SQ_INSTS_VALU']:.1f} % of it, {v/l if l else 0:.2f} VALU per LDS")


def main(logdir):
    print("# gfx1013 hardware counter summary\n")
    validation(logdir)
    gemms(logdir)
    scalarisation(logdir)
    alu_cost(logdir)
    llama(logdir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "logs/hw-counters-2026-09-25")
