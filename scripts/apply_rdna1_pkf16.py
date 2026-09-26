#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Route RDNA1 prefill matmuls to the packed-fp16 tile GEMM (mmf16-rdna1.cu) instead of MMQ.
Usage: apply_rdna1_pkf16.py <llama.cpp root>"""
import sys, pathlib
root = pathlib.Path(sys.argv[1]); p = root / "ggml/src/ggml-cuda/ggml-cuda.cu"; s = p.read_text()
if "mmf16-rdna1.cuh" not in s:
    anchor = '#include "ggml-cuda/mmq.cuh"\n'
    assert anchor in s, "include anchor not found"
    s = s.replace(anchor, anchor + '#include "ggml-cuda/mmf16-rdna1.cuh"\n', 1)
old = "    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {"
new = """    // RDNA1 has no int8 dot product, so MMQ runs the slowest arithmetic on the chip; a packed-fp16
    // tile GEMM is 2.2 to 2.4 times faster for prefill (logs/alu-rates-2026-09-19)
    if (ggml_cuda_mmf16_rdna1_supported(src0, src1, dst, cc)) {
        ggml_cuda_mmf16_rdna1(ctx, src0, src1, dst);
        return;
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {"""
assert s.count(old) == 1, "dispatch site not found"
s = s.replace(old, new, 1); p.write_text(s); print("ggml-cuda.cu: RDNA1 prefill routed to the packed-fp16 GEMM")
