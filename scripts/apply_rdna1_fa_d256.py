#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""D=256 flash-attention tile on RDNA1: set the ncols=16 and ncols=32 rows of the RDNA1 table and add a
dispatch cap that keeps D=256 prefill on the 16-column tile (its 8x2 / 2x8 instances compile without
spill at 512 threads, occupancy 2, nbatch_K 64, where every 32-column instance spills). The cap reads
GGML_FA_D256_CAP16 (default 1) so one build measures both. Usage:
  apply_rdna1_fa_d256.py <ggml-cuda dir> <row16 nthreads:occ:nbatch_fa:nbatch_K> <row32 ...> [cap default 1|0]"""
import sys, pathlib, re
d = pathlib.Path(sys.argv[1]); r16, r32 = sys.argv[2], sys.argv[3]; capdef = sys.argv[4] if len(sys.argv) > 4 else "1"
p = d / "fattn-tile.cuh"; s = p.read_text()
def setrow(s, ncols, spec):
    pat = re.compile(r"    GGML_CUDA_FATTN_TILE_CONFIG_CASE\(256, 256, %d,[^)]*\)\n" % ncols)
    line = "    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, %d, %s)\n" % (ncols, ", ".join(spec.split(":")))
    i = s.index("ggml_cuda_fattn_tile_get_config_amd_rdna1"); j = s.index("return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);", i)
    m = pat.search(s, i, j); assert m, "no D=256 row for ncols %d" % ncols
    return s[:m.start()] + line + s[m.end():]
s = setrow(s, 16, r16); s = setrow(s, 32, r32)
old = """    {
        if (Q->ne[1] > 16/ncols2) {
            constexpr int cols_per_block = 32;"""
new = """    {
        // RDNA1, D=256: the 32-column tile spills at every geometry tried, the 16-column one does not
        static const bool rdna1_d256_cap16 = [] { const char * e = getenv("GGML_FA_D256_CAP16"); return e ? atoi(e) != 0 : CAPDEF; }();
        const bool cap16 = GGML_CUDA_CC_IS_RDNA1(cc) && DKQ == 256 && rdna1_d256_cap16;
        if (!cap16 && Q->ne[1] > 16/ncols2) {
            constexpr int cols_per_block = 32;"""
if new not in s:
    assert old in s; s = s.replace(old, new.replace("CAPDEF", "true" if capdef == "1" else "false"), 1)
if "#include <cstdlib>" not in s: s = s.replace('#include "common.cuh"\n', '#include "common.cuh"\n#include <cstdlib>\n', 1)
p.write_text(s); print("fattn-tile.cuh: D=256 row16 %s row32 %s, cap16 default %s (GGML_FA_D256_CAP16)" % (r16, r32, capdef))
