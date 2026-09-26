#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Final form of experiment J: the RDNA1 D=256 ncols-32 tile row becomes 256 threads, occupancy 2,
nbatch_fa 32, nbatch_K 64, which takes the MoE's 4x8 prefill instance from 52 spilled registers to 5 and
halves its 2048-token attention op (the 27B's 16x2 loses 42 percent). No run-time switch.
Usage: apply_rdna1_fa_d256_row.py <ggml-cuda dir>"""
import sys, pathlib, re
p = pathlib.Path(sys.argv[1]) / "fattn-tile.cuh"; s = p.read_text()
i = s.index("ggml_cuda_fattn_tile_get_config_amd_rdna1"); j = s.index("return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);", i)
pat = re.compile(r"    GGML_CUDA_FATTN_TILE_CONFIG_CASE\(256, 256, 32,[^)]*\)\n")
m = pat.search(s, i, j); assert m, "no D=256 ncols-32 row"
line = "    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2, 32, 64)\n"
if s[m.start():m.end()] == line: print("already applied"); sys.exit(0)
s = s[:m.start()] + line + s[m.end():]; p.write_text(s); print("fattn-tile.cuh: D=256 ncols-32 row -> 256, 2, 32, 64")
