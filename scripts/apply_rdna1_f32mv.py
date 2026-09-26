#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Dispatch hook: single-column q4_K matvec on RDNA1, fused or not (bias, gate, GLU), goes to the
# float-activation kernel in mmvq-rdna1-f32.cu instead of quantizing the activations to q8_1 and running
# the emulated int8 dot. Placed after the fusion pointers are resolved, before the quantization.
# Usage: apply_rdna1_f32mv.py <tree>
import sys
p = sys.argv[1] + "/ggml/src/ggml-cuda/mmvq.cu"; s = open(p).read()
assert "mmvq-rdna1-f32.cuh" not in s, "already applied"
s = s.replace('#include "vecdotq.cuh"\n', '#include "vecdotq.cuh"\n#include "mmvq-rdna1-f32.cuh"\n', 1)
anchor = "    quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);\n"
assert s.count(anchor) == 1
hook = """    // RDNA1: the emulated int8 dot costs 1.9x the instructions of a native one; float activations avoid it
    if (!ids && (!fusion || (fusion->x_scale == nullptr && fusion->gate_scale == nullptr))) {
        const int cc_f32 = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        if (ggml_cuda_mmvq_rdna1_f32_supported(src0, src1, dst, cc_f32)) {
            ggml_cuda_mmvq_rdna1_f32(ctx, src0, src1, dst, fusion ? &fusion_local : nullptr);
            return;
        }
    }
"""
s = s.replace(anchor, hook + anchor, 1); open(p, "w").write(s); print("mmvq.cu: float-activation hook (fused) added")
