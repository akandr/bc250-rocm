#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""RDNA1: one-token flash attention goes to the vector kernel when the tile kernel would run with
ncols2 = 1, that is when the GQA optimisation applies but the head ratio has no power-of-two factor
(deepseek-r1-14B and qwen3-14B, 40 heads over 8). Without matrix cores the chooser sends such an op to a
1 x 1 tile that spills; the vector kernel measured 2.9 times faster on the op and 42 percent on decode
at a 4096-token depth. Ratios with a power-of-two part (8B: 4, 1.5B: 6, the D=256 models) stay on the
tile, which measured faster there. Usage: apply_rdna1_fa_vec_decode.py <ggml-cuda dir>"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "fattn.cu"; s = p.read_text()
old = """    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }"""
new = """    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                // RDNA1: a head ratio with no power-of-two factor would give the tile kernel ncols2 = 1,
                // and that instance spills; the vector kernel is 2.9x faster there (BC-250 repository)
                const bool rdna1_no_gqa_tile = GGML_CUDA_CC_IS_RDNA1(cc) && gqa_ratio_eff == 1;
                if (!gqa_opt_applies || rdna1_no_gqa_tile) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }"""
if new in s: print("already applied"); sys.exit(0)
assert old in s, "chooser text not found"
s = s.replace(old, new, 1)
p.write_text(s); print("fattn.cu: RDNA1 one-token flash attention with ncols2 == 1 -> vector kernel")
