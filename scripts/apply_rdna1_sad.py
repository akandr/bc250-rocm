#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# RDNA1 has no v_dot4, so ggml_cuda_dp4a(0x01010101, u, acc), the sum of four int8 activation
# values, costs the full six-instruction emulation. v_sad_u8 sums four unsigned bytes in one
# instruction; XOR with 0x80808080 turns two's-complement bytes into biased unsigned ones, and the
# bias comes off once. Gated on RDNA1; every other target keeps the dp4a form.
# Usage: apply_rdna1_sad.py <path to ggml-cuda dir>
import sys, re
d = sys.argv[1]
p = d + "/vecdotq.cuh"; s = open(p).read()
helper = '''
// Sum of the four int8 values packed in u, plus acc. On RDNA1 the dp4a emulation costs six
// instructions per word; v_sad_u8 on sign-biased bytes costs two, and the bias comes off once.
static __device__ __forceinline__ int ggml_cuda_sum_i8x4(const int u, const int acc) {
#if defined(GGML_USE_HIP) && defined(RDNA1)
    return (int) __builtin_amdgcn_sad_u8((unsigned) (u ^ 0x80808080), 0u, (unsigned) acc) - 512;
#else
    return ggml_cuda_dp4a(0x01010101, u, acc);
#endif
}
'''
anchor = "#define VDR_Q4_0_Q8_1_MMVQ"
assert s.count(anchor) == 1 and "ggml_cuda_sum_i8x4" not in s
s = s.replace(anchor, helper.lstrip("\n") + "\n" + anchor, 1)
subs = [
    ("const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0)); // sum of u",
     "const int dot2 = ggml_cuda_sum_i8x4(u[2*i+1], ggml_cuda_sum_i8x4(u[2*i+0], 0)); // sum of u"),
    ("const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0)); // sum of u",
     "const int dot2 = ggml_cuda_sum_i8x4(u[2*i+0], ggml_cuda_sum_i8x4(u[2*i+1], 0)); // sum of u"),
    ("sumi_m0 = ggml_cuda_dp4a(0x01010101, u[i], sumi_m0);", "sumi_m0 = ggml_cuda_sum_i8x4(u[i], sumi_m0);"),
    ("sumi_m1 = ggml_cuda_dp4a(0x01010101, u[i], sumi_m1);", "sumi_m1 = ggml_cuda_sum_i8x4(u[i], sumi_m1);"),
]
n = 0
for a, b in subs:
    c = s.count(a); assert c >= 1, a
    s = s.replace(a, b); n += c
open(p, "w").write(s)
print(f"vecdotq.cuh: helper added, {n} sites switched")
