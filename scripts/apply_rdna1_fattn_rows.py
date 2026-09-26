#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Extend the RDNA1 flash-attention tile table in ggml/src/ggml-cuda/fattn-tile.cuh (added by
# patch 0004) with rows for D=256 and D=512, chosen by compile-time register sweeps on gfx1013
# (logs/rdna1-fattn-spill-2026-09-17/kernel-resource-usage.log).
#
# Usage: apply_rdna1_fattn_rows.py <llama.cpp tree> [D512 rows "ncols:nthreads:occ:nbatch_fa:nbatch_K" ...]
#                                   [--d128 "ncols:nthreads:occ:nbatch_fa:nbatch_K" ...]
# --d128 rewrites the D=128 rows patch 4 already carries (rows 32 and 64), or adds rows it does not.
# Idempotent: re-running replaces the block it wrote before.
import sys, re

args = sys.argv[1:]
tree = args.pop(0)
d128 = []
if "--d128" in args:
    k = args.index("--d128"); d128 = args[k + 1:]; args = args[:k]
d512 = args or ["32:512:2:32:64", "16:512:2:32:64"]
p = f"{tree}/ggml/src/ggml-cuda/fattn-tile.cuh"
s = open(p).read()

BEGIN = "    // D=256 and D=512 rows, from the gfx1013 register sweeps (BC-250 repository)\n"
END = "    // end of D=256 and D=512 rows\n"
row = lambda D, spec: "    GGML_CUDA_FATTN_TILE_CONFIG_CASE(%s, %s, %s, %s, %s, %s, %s)\n" % ((D, D) + tuple(spec.split(":")))
d256 = ["32:512:3:32:128", "16:512:3:32:128", "8:256:4:32:64", "4:128:4:32:64", "2:64:4:32:64"]
block = BEGIN + "".join(row(256, x) for x in d256) + "\n" + "".join(row(512, x) for x in d512) + END

i = s.index("ggml_cuda_fattn_tile_get_config_amd_rdna1")
if BEGIN in s:
    a = s.index(BEGIN); b = s.index(END) + len(END)
    s = s[:a] + block + s[b:]
    print("replaced existing block")
else:
    j = s.index("    return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);", i)
    s = s[:j] + block + "\n" + s[j:]
    print("inserted block")
for spec in d128:
    ncols, nt, occ, nfa, nk = spec.split(":")
    pat = re.compile(r"    GGML_CUDA_FATTN_TILE_CONFIG_CASE\(128, 128,\s*%s,[^)]*\)\n" % ncols)
    line = "    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, %s, %s, %s, %s, %s)\n" % (ncols, nt, occ, nfa, nk)
    i = s.index("ggml_cuda_fattn_tile_get_config_amd_rdna1")
    m = pat.search(s, i)
    if m and m.start() < s.index(BEGIN):
        s = s[:m.start()] + line + s[m.end():]; print("D=128 row rewritten:", spec)
    else:
        s = s[:s.index(BEGIN)] + line + s[s.index(BEGIN):]; print("D=128 row added:", spec)
open(p, "w").write(s)
print("D=512 rows:", d512)
