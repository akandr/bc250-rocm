#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# A long-K variant of the RDNA1 matrix-vector entry: one wave per row was measured 3 to 6 percent slower
# than four cooperating warps on the K = 8960 ffn_down shapes, and faster everywhere shorter. Mirrors the
# existing small_k template flag: the host sets long_k when K >= 8192 on the RDNA1 table, and calc_nwarps
# returns 4 for it. Applies on top of apply_rdna1_mmvq_table.py. Usage: apply_rdna1_longk.py <tree>
import sys
p = sys.argv[1] + "/ggml/src/ggml-cuda/mmvq.cu"; s = open(p).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:70])
    s = s.replace(a, b)
rep("static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id) {",
    "static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool long_k = false) {")
rep("""                case GGML_TYPE_Q8_0:
                    return 4;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {""",
    """                case GGML_TYPE_Q8_0:
                    return 4;
                default:
                    return long_k ? 4 : 1;   // K >= 8192: four warps split the long row
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {""")
rep("template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false>\n__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id())*ggml_cuda_get_physical_warp_size(), 1)",
    "template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool long_k = false>\n__launch_bounds__(calc_nwarps(type, ncols_dst, get_device_table_id(), long_k)*ggml_cuda_get_physical_warp_size(), 1)")
rep("    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id);\n    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);",
    "    constexpr int nwarps = calc_nwarps(type, ncols_dst, table_id, long_k);\n    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);")
rep("        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false) {\n    const int nwarps = calc_nwarps(type, ncols_dst, table_id);",
    "        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool long_k = false) {\n    const int nwarps = calc_nwarps(type, ncols_dst, table_id, long_k);")
rep("template<ggml_type type, int c_ncols_dst, bool small_k = false>\nstatic void mul_mat_vec_q_switch_fusion(",
    "template<ggml_type type, int c_ncols_dst, bool small_k = false, bool long_k = false>\nstatic void mul_mat_vec_q_switch_fusion(")
rep("ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k>, launch_params,",
    "ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, long_k>, launch_params,")
rep("ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k>, launch_params,",
    "ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, long_k>, launch_params,")
rep("""            bool use_small_k = should_use_small_k(c_ncols_dst);

            if (use_small_k) {""",
    """            bool use_small_k = should_use_small_k(c_ncols_dst);
            // RDNA1: a long row is better split across four warps than walked by one wave
            const bool use_long_k = table_id == MMVQ_PARAMETERS_RDNA1 && !use_small_k && ncols_x >= 8192;

            if (use_long_k) {
                std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                        nsamples_dst, warp_size, table_id, false, true);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, true>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            } else if (use_small_k) {""")
open(p, "w").write(s); print("mmvq.cu: long_k plumbed")
