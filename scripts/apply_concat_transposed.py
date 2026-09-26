#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""CONCAT along dim 0 with a transposed second source (the recurrent conv input in Mamba/GDN graphs is
ggml_concat(state, ggml_transpose(x), 0)): the generic non-contiguous kernel reads x one element per lane
40 KB apart, 15 GB/s on this GPU. A shared-memory tile transposes 32x32 blocks so both the read and the
write are coalesced. Also adds a test-backend-ops case with a transposed b (v bit 16).
Usage: apply_concat_transposed.py <llama.cpp root>"""
import sys, pathlib
root = pathlib.Path(sys.argv[1])
p = root / "ggml/src/ggml-cuda/concat.cu"; s = p.read_text()
if "concat_dim0_transposed" not in s:
    kern = '''
// dim 0 with a transposed second source (nb10 > nb11 == element size): the generic kernel above reads
// src1 one element per lane at stride nb10, which is uncoalesced by the row length. Tile it through
// shared memory so the read walks src1's contiguous dimension and the write walks dst's.
template <typename T>
static __global__ void concat_dim0_transposed(const char * src1, char * dst,
        int64_t ne10, int64_t ne11, int64_t ne2, uint64_t nb10, uint64_t nb11, uint64_t nb12, uint64_t nb13,
        int64_t ne00, uint64_t nb1, uint64_t nb2, uint64_t nb3) {
    __shared__ T tile[32][33];
    const int64_t i0_base = (int64_t) blockIdx.x * 32;     // along src1's dim 0, dst i0 = ne00 + i0
    const int64_t i1_base = (int64_t) blockIdx.y * 32;     // along dim 1
    const int64_t i2 = blockIdx.z % ne2, i3 = blockIdx.z / ne2;
    const char * s = src1 + i2 * nb12 + i3 * nb13;
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        const int64_t i0 = i0_base + j, i1 = i1_base + threadIdx.x;   // consecutive lanes: consecutive i1, contiguous in src1
        if (i0 < ne10 && i1 < ne11) tile[j][threadIdx.x] = *(const T *) (s + i0 * nb10 + i1 * nb11);
    }
    __syncthreads();
    char * d = dst + i2 * nb2 + i3 * nb3;
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        const int64_t i1 = i1_base + j, i0 = i0_base + threadIdx.x;   // consecutive lanes: consecutive i0, contiguous in dst
        if (i0 < ne10 && i1 < ne11) *(T *) (d + i1 * nb1 + (ne00 + i0) * sizeof(T)) = tile[threadIdx.x][j];
    }
}

// the first source's part of the same dst rows, ne00 elements per row, any strides
template <typename T>
static __global__ void concat_dim0_src0_part(const char * src0, char * dst, int64_t ne00,
        uint64_t nb00, uint64_t nb01, uint64_t nb02, uint64_t nb03, uint64_t nb1, uint64_t nb2, uint64_t nb3) {
    const int64_t i1 = blockIdx.x, i2 = blockIdx.y, i3 = blockIdx.z;
    for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
        *(T *) (dst + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * sizeof(T)) =
            *(const T *) (src0 + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
    }
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {'''
    s = s.replace('''
template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {''', kern, 1)
    old = '''    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);'''
    new = '''    } else if (dim == 0 && !ggml_is_quantized(src0->type) && dst->nb[0] == sizeof(T) &&
               src1->nb[1] == sizeof(T) && src1->nb[0] > sizeof(T) && src0->ne[1] == src1->ne[1] &&
               src0->ne[2] == src1->ne[2] && src0->ne[3] == src1->ne[3] && dst->ne[2] * dst->ne[3] <= 65535) {
        const dim3 grid_t((src1->ne[0] + 31) / 32, (src1->ne[1] + 31) / 32, dst->ne[2] * dst->ne[3]);
        concat_dim0_transposed<T><<<grid_t, dim3(32, 8, 1), 0, stream>>>(
            (const char *) src1->data, (char *) dst->data, src1->ne[0], src1->ne[1], dst->ne[2],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], src0->ne[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        const dim3 grid_s(dst->ne[1], dst->ne[2], dst->ne[3]);
        concat_dim0_src0_part<T><<<grid_s, 32, 0, stream>>>(
            (const char *) src0->data, (char *) dst->data, src0->ne[0],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3], dst->nb[1], dst->nb[2], dst->nb[3]);
    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);'''
    assert old in s; s = s.replace(old, new, 1)
    p.write_text(s); print("concat.cu: transposed-source dim-0 path added")
else:
    print("concat.cu: already applied")
t = root / "tests/test-backend-ops.cpp"; s = t.read_text()
if "v & 16" not in s:
    old = '''        } else if (v & 8) {
            auto ne = ne_b; ne[2] *= 3; ne[3] *= 2;
            b = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(b, "b");

            b = ggml_view_4d(ctx, b, ne_b[0], ne_b[1], ne_b[2], ne_b[3], b->nb[1], b->nb[2], b->nb[3], 0);
            ggml_set_name(b, "view_of_b");
        } else {'''
    new = '''        } else if (v & 8) {
            auto ne = ne_b; ne[2] *= 3; ne[3] *= 2;
            b = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(b, "b");

            b = ggml_view_4d(ctx, b, ne_b[0], ne_b[1], ne_b[2], ne_b[3], b->nb[1], b->nb[2], b->nb[3], 0);
            ggml_set_name(b, "view_of_b");
        } else if (v & 16) {
            // b transposed in its first two dimensions, as the recurrent conv input is built
            std::array<int64_t, 4> ne = {ne_b[1], ne_b[0], ne_b[2], ne_b[3]};
            b = ggml_new_tensor(ctx, type, 4, ne.data());
            ggml_set_name(b, "b");

            b = ggml_transpose(ctx, b);
            ggml_set_name(b, "transpose_of_b");
        } else {'''
    assert old in s; s = s.replace(old, new, 1)
    old = '''            test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {11, 12, 13, 14}, 7, dim, v));'''
    new = '''            test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {11, 12, 13, 14}, 7, dim, v));
            if (dim == 0 && v == 0) {
                test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {11, 12, 13, 14}, 7, 0, 16));
                test_cases.emplace_back(new test_concat(GGML_TYPE_F16, {11, 12, 13, 14}, 7, 0, 16));
                test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {3, 10240, 1, 1}, 2048, 0, 16));
                test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {3, 8192, 2, 1}, 37, 0, 16));
            }'''
    assert s.count(old) == 1; s = s.replace(old, new, 1)
    t.write_text(s); print("test-backend-ops.cpp: transposed-b concat cases added")
else:
    print("test-backend-ops.cpp: already applied")
