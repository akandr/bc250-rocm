#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Extend the RDNA1 float-activation matvec hook in mmvq.cu to MUL_MAT_ID (one token): pass the ids tensor
through and drop the !ids guard. The kernel files themselves are copied alongside. Usage: apply_rdna1_f32iq.py <ggml-cuda dir>"""
import sys, pathlib
d = pathlib.Path(sys.argv[1]); p = d / "mmvq.cu"; s = p.read_text()
old1 = "if (!ids && (!fusion || (fusion->x_scale == nullptr && fusion->gate_scale == nullptr))) {"
new1 = "if (!fusion || (fusion->x_scale == nullptr && fusion->gate_scale == nullptr)) {"
old2 = "ggml_cuda_mmvq_rdna1_f32_supported(src0, src1, dst, cc_f32)"
new2 = "ggml_cuda_mmvq_rdna1_f32_supported(src0, src1, ids, dst, cc_f32)"
old3 = "ggml_cuda_mmvq_rdna1_f32(ctx, src0, src1, dst, fusion ? &fusion_local : nullptr);"
new3 = "ggml_cuda_mmvq_rdna1_f32(ctx, src0, src1, ids, dst, fusion ? &fusion_local : nullptr);"
for o, n in ((old1, new1), (old2, new2), (old3, new3)):
    if n in s: print("already:", n[:50]); continue
    assert s.count(o) == 1, o
    s = s.replace(o, n)
p.write_text(s); print("mmvq.cu hook: ids passed through")
