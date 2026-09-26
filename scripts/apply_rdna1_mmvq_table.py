#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Give RDNA1 its own MMVQ parameter table instead of borrowing RDNA2's: one wave per row for the
# K-quants and IQ types (measured: 1.5B +26 percent, 14B +14, MoE +34 to 45), the generic four-warp
# form for the simple-vec_dot types, where one wave per row lost 4 percent on the 8B Q8_0.
# Applies on top of the RDNA1 -> RDNA2 routing (replaces it). Usage: apply_rdna1_mmvq_table.py <tree>
import sys
p = sys.argv[1] + "/ggml/src/ggml-cuda/mmvq.cu"; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b, 1)
rep("    MMVQ_PARAMETERS_RDNA4\n};", "    MMVQ_PARAMETERS_RDNA4,\n    MMVQ_PARAMETERS_RDNA1\n};")
rep("#elif defined(RDNA2) || defined(RDNA3_5) || defined(RDNA1)\n    return MMVQ_PARAMETERS_RDNA2;",
    "#elif defined(RDNA2) || defined(RDNA3_5)\n    return MMVQ_PARAMETERS_RDNA2;\n#elif defined(RDNA1)\n    return MMVQ_PARAMETERS_RDNA1;")
rep("    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc) || GGML_CUDA_CC_IS_RDNA1(cc)) {\n        return MMVQ_PARAMETERS_RDNA2;\n    }",
    "    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {\n        return MMVQ_PARAMETERS_RDNA2;\n    }\n    if (GGML_CUDA_CC_IS_RDNA1(cc)) {\n        return MMVQ_PARAMETERS_RDNA1;\n    }")
rep("    if (table_id == MMVQ_PARAMETERS_RDNA4) {\n        // nwarps=8 benefits",
    """    if (table_id == MMVQ_PARAMETERS_RDNA1) {
        // RDNA1 (gfx1010, gfx1013): one wave per row suits the K-quants and IQ types, whose vec_dot is
        // long enough to keep a wave busy; the simple types are memory-bound and keep the generic
        // four warps per row, which measured 4 percent faster on Q8_0.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 4;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits""")
open(p, "w").write(s); print("mmvq.cu: RDNA1 table added")
