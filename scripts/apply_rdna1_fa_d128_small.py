#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""RDNA1 flash-attention tile rows for D=128 at 4 and 2 columns, the decode-shaped instances with a GQA
ratio of 4 (1x4: qwen3-8B) and 2 (1x2: qwen2.5-1.5B). The RDNA1 table had rows for 8 columns and up, so
these instances took the shared RDNA row and the 1x4 spilled 240 registers; 128 threads, occupancy 4,
nbatch_fa 32, nbatch_K 64 compiles the 1x4 clean at 121 registers, occupancy 8, and 64 threads with the
same batching the 1x2 at 147 (sweep128-c4.log, sweep128-c2.log). Usage: apply_rdna1_fa_d128_small.py <ggml-cuda dir>"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "fattn-tile.cuh"; s = p.read_text()
anchor = "    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 8, 256, 4, 32, 64)\n"
rows = anchor + "    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 4, 128, 4, 32, 64)\n    GGML_CUDA_FATTN_TILE_CONFIG_CASE(128, 128, 2,  64, 4, 32, 64)\n"
if "CONFIG_CASE(128, 128, 4, 128, 4, 32, 64)" in s: print("already applied"); sys.exit(0)
assert anchor in s; s = s.replace(anchor, rows, 1); p.write_text(s); print("fattn-tile.cuh: D=128 rows for 4 and 2 columns added")
