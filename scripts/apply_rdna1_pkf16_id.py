#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Route RDNA1 MUL_MAT_ID prefill to the packed-fp16 tile GEMM's expert path.
Apply after apply_rdna1_pkf16.py. Usage: apply_rdna1_pkf16_id.py <llama.cpp root>"""
import sys, pathlib
root = pathlib.Path(sys.argv[1]); p = root / "ggml/src/ggml-cuda/ggml-cuda.cu"; s = p.read_text()
assert "mmf16-rdna1.cuh" in s, "apply apply_rdna1_pkf16.py first"
old = "        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {"
new = """        // the experts are 60 percent of a mixture-of-experts model's prefill and MMQ's emulated int8
        // is the slowest arithmetic on RDNA1 (logs/moe-decode-2026-09-20)
        if (ggml_cuda_mmf16_rdna1_id_supported(src0, src1, ids, dst, cc)) {
            ggml_cuda_mmf16_rdna1_id(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {"""
assert s.count(old) == 1, "MUL_MAT_ID dispatch site not found"
s = s.replace(old, new, 1); p.write_text(s); print("ggml-cuda.cu: RDNA1 expert prefill routed to the packed-fp16 GEMM")
