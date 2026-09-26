#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Instruction mix per quant type in the RDNA1 float matvec ISA.

Usage: hipcc -O3 --offload-arch=gfx1013 -S --cuda-device-only mmvq-rdna1-f32.cu -o mv.s
       isa_mix.py mv.s

Each row is one kernel instantiation, identified by the ggml_type in its mangled name; the two rows per
type are the variants with and without bias/gate/GLU fusion. Counts are whole-kernel, so they include
prologue, epilogue and the 64-bit address arithmetic every variant shares: compare types, not absolutes.
"""
import re
import sys

src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mv.s"
s = open(src).read()

TYPE_NAMES = {
    2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0",
    10: "q2_K", 11: "q3_K", 12: "q4_K", 13: "q5_K", 14: "q6_K",
    16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs", 19: "iq1_s",
    20: "iq4_nl", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs",
}


def bucket(ins):
    if ins.startswith(("v_fma", "v_mac", "v_fmac", "v_pk_fma", "v_dot", "v_mad")):
        return "fma"
    if "cvt" in ins:
        return "cvt"
    if ins.startswith(("v_xor", "s_xor", "v_and", "s_and", "v_or", "s_or", "v_not", "v_bfi", "v_bfe", "s_bfe")):
        return "bitwise"
    if "lshl" in ins or "lshr" in ins or "ashr" in ins or "perm" in ins or "pack" in ins or "sdwa" in ins:
        return "shift/pack"
    if ins.startswith(("global_load", "buffer_load", "flat_load")):
        return "vmem load"
    if ins.startswith("s_load"):
        return "smem load"
    if ins.startswith("ds_"):
        return "lds"
    if ins.startswith(("global_store", "buffer_store", "flat_store")):
        return "store"
    if ins.startswith(("v_mul", "v_add", "v_sub", "v_max", "v_min", "v_rcp", "v_cmp", "v_cndmask", "v_mov")):
        return "other valu"
    if ins.startswith("v_"):
        return "other valu"
    return "scalar/ctrl"


COLS = ["fma", "cvt", "bitwise", "shift/pack", "other valu", "vmem load", "smem load", "lds", "store", "scalar/ctrl"]

rows = []
for m in re.finditer(r"^([A-Za-z_][\w.$]*):\s*;+\s*@", s, re.M):
    name = m.group(1)
    if "mul_mat_vec" not in name:
        continue
    seg = s[m.end():]
    e = re.search(r"^\.Lfunc_end", seg, re.M)
    seg = seg[:e.start()] if e else seg
    ops = {}
    for ins in re.findall(r"^\s+([a-z][a-z0-9_]*)", seg, re.M):
        if not ins.startswith(("v_", "s_", "ds_", "global_", "buffer_", "flat_")):
            continue
        k = bucket(ins)
        ops[k] = ops.get(k, 0) + 1
    ty = re.search(r"IL9ggml_type(\d+)E", name)
    tyn = int(ty.group(1)) if ty else -1
    rows.append((TYPE_NAMES.get(tyn, str(tyn)), sum(ops.values()), ops))

rows.sort(key=lambda r: -r[1])
hdr = f"{'type':>9s} {'total':>6s} " + " ".join(f"{c:>11s}" for c in COLS)
print(hdr)
print("-" * len(hdr))
for name, tot, ops in rows:
    print(f"{name:>9s} {tot:6d} " + " ".join(f"{ops.get(c, 0):11d}" for c in COLS))
