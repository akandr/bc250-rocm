#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""GATED_DELTA_NET on RDNA1: lanes per column as a template parameter. The kernel gives every column of the
S_v x S_v state one full wave (32 lanes, 4 rows each at S_v = 128) and reduces across the wave twice per
token; ggml-vulkan's shader uses 8 lanes per column for S_v >= 128 (four columns per subgroup, 16 rows per
lane). This adds the parameter, keeps the existing geometry everywhere, and on RDNA1 picks the lane count
from GGML_GDN_LANES for the S_v = 128 kernels (experiment build; the final form hardcodes the winner).
Usage: apply_rdna1_gdn_lanes.py <ggml-cuda dir>"""
import sys, pathlib, re
d = pathlib.Path(sys.argv[1])
p = d / "gated_delta_net.cu"; s = p.read_text()
if "int lanes_t" in s: print("already applied"); sys.exit(0)

# 1. template parameter and geometry
s = s.replace("template <int S_v, bool KDA, bool keep_rs_t>\n__global__ void __launch_bounds__",
              "template <int S_v, bool KDA, bool keep_rs_t, int lanes_t = 0>\n__global__ void __launch_bounds__", 1)
old = """    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;
"""
new = """    // each group of `lanes` threads owns one column, using warp-level primitives to reduce across rows;
    // lanes_t == 0 keeps one column per warp
    constexpr int  phys_warp = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    constexpr int  lanes     = lanes_t > 0 ? lanes_t : phys_warp;
    constexpr int  cols_per_warp = phys_warp / lanes;
    const int      lane     = threadIdx.x % lanes;
    const int      col      = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + threadIdx.x / lanes;
"""
assert old in s; s = s.replace(old, new, 1)
old = """    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
"""
new = """    constexpr int warp_size = lanes;   // the reduction width: the lanes that share a column
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of the lanes per column");
    static_assert(phys_warp % lanes == 0, "lanes per column must divide the wave");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
"""
assert old in s; s = s.replace(old, new, 1)

# 2. host: grid z covers S_v / (num_warps * cols_per_warp) with the RDNA1 lane choice for S_v = 128
old = """    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);
"""
new = """    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    // RDNA1, more than one token: fewer lanes per column, more columns per wave (ggml-vulkan uses 8 for
    // S_v >= 128); at one token the one-column-per-wave form is the faster one
    static const int rdna1_lanes = [] { const char * e = getenv("GGML_GDN_LANES"); return e ? atoi(e) : 8; }();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int lanes128 = (GGML_CUDA_CC_IS_RDNA1(cc) && S_v == 128 && warp_size == 32 && n_tokens > 1) ? rdna1_lanes : 0;
    const int cols_per_warp = lanes128 > 0 ? warp_size / lanes128 : 1;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps * cols_per_warp - 1) / (num_warps * cols_per_warp));
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);
"""
assert old in s; s = s.replace(old, new, 1)
old = """        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            break;
        }
"""
new = """        case 128: {
#define GDN_LAUNCH_128(L) ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, L>, launch_params, \\
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, \\
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K)
            switch (lanes128) {
                case 4:  GDN_LAUNCH_128(4);  break;
                case 8:  GDN_LAUNCH_128(8);  break;
                case 16: GDN_LAUNCH_128(16); break;
                default: GDN_LAUNCH_128(0);  break;
            }
#undef GDN_LAUNCH_128
            break;
        }
"""
assert old in s; s = s.replace(old, new, 1)
if "#include <cstdlib>" not in s: s = s.replace('#include "ggml-cuda/common.cuh"\n', '#include "ggml-cuda/common.cuh"\n#include <cstdlib>\n', 1)
p.write_text(s); print("gated_delta_net.cu: lanes per column parameterised; RDNA1 prefill 8 (GGML_GDN_LANES overrides)")
