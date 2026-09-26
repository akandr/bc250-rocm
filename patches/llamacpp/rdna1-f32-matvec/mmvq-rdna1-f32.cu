// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// Float-activation matrix-vector products for RDNA1: q4_K, q5_K, q6_K, q8_0, iq2_xxs, iq3_xxs, iq3_s.
//
// RDNA1 has no v_dot4, so MMVQ's int8 dot products cost 1.9 times the instructions they cost on RDNA2
// (logs/op-perf-hip-vs-vulkan-2026-09-17/mmvq-isa.log). Vulkan's matvec on the same GPU does not quantize
// the activations at all: it unpacks the weights to float and runs fma chains against float activations,
// and it is 1.5 to 1.9 times faster per shape. This is that formulation in HIP, laid out like
// ggml-vulkan's mul_mat_vec_*.comp: 16 threads share a 256-value superblock, each owning 16 values; one
// 32-lane wave covers two superblocks per iteration and two rows per block. Single column only; with an
// expert-id tensor (MUL_MAT_ID, one token) each block's channel is one used expert.
#include "common.cuh"
#include <type_traits>
#include "unary.cuh"
#include "mmvq-rdna1-f32.cuh"

#define MMVQ_F32_ROWS 2

static __device__ __forceinline__ float4 ld4(const float * p) { return *reinterpret_cast<const float4 *>(p); }

static __device__ __forceinline__ float4 unpack_u8x4(const uint32_t v) {
    return make_float4((float) ( v        & 0xFF), (float) ((v >>  8) & 0xFF),
                       (float) ((v >> 16) & 0xFF), (float) ((v >> 24) & 0xFF));
}

static __device__ __forceinline__ float dot4(const float4 a, const float4 b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w)));
}

static __device__ __forceinline__ float sum4(const float4 a) { return a.x + a.y + a.z + a.w; }

// Apply four sign bits (bit `bit`..`bit+3` of `signs`) to the four grid values.
static __device__ __forceinline__ float4 signed4(const float4 g, const uint32_t signs, const int bit) {
    return make_float4(__uint_as_float(__float_as_uint(g.x) ^ (((signs >> (bit + 0)) & 1u) << 31)),
                       __uint_as_float(__float_as_uint(g.y) ^ (((signs >> (bit + 1)) & 1u) << 31)),
                       __uint_as_float(__float_as_uint(g.z) ^ (((signs >> (bit + 2)) & 1u) << 31)),
                       __uint_as_float(__float_as_uint(g.w) ^ (((signs >> (bit + 3)) & 1u) << 31)));
}

// q4_K and q5_K share the superblock layout (dm, 12 scale bytes, 128 nibble bytes); q5_K adds 32 bytes of
// fifth bits. One superblock of one row against 16 activations held by this thread; returns d*sum - dmin*smin.
template <typename block_t, bool has_qh>
static __device__ __forceinline__ float q45_K_superblock_dot(
        const block_t * b, const int v_im, const int q_offset,
        const float4 by10, const float4 by132, const float4 by20, const float4 by232) {
    const float2 dm = __half22float2(b->dm);
    const uint16_t * sc16 = (const uint16_t *) b->scales;
    const uint32_t scale0 = sc16[v_im], scale4 = sc16[v_im + 2], scale8 = sc16[v_im + 4];
    const uint32_t s04l = (scale4 << 16) | scale0;
    const uint32_t s04h = (s04l & 0xC0C0C0C0u) >> 2;
    const float4 sc_lo = unpack_u8x4(s04l & 0x3F3F3F3Fu);                                  // sc0..sc3
    const float4 sc_hi = unpack_u8x4((((scale8 << 12) | scale8) & 0x0F0F0F0Fu) | s04h);     // sc4..sc7
    const uint32_t * q32 = (const uint32_t *) b->qs;
    const uint32_t qs0  = q32[q_offset / 4];
    const uint32_t qs64 = q32[q_offset / 4 + 16];
    uint32_t q0lo  =  qs0        & 0x0F0F0F0Fu;
    uint32_t q0hi  = (qs0  >> 4) & 0x0F0F0F0Fu;
    uint32_t q64lo =  qs64       & 0x0F0F0F0Fu;
    uint32_t q64hi = (qs64 >> 4) & 0x0F0F0F0Fu;
    if constexpr (has_qh) {
        const uint32_t qh = *(const uint32_t *) (b->qh + (q_offset & 31));
        q0lo  |= ((qh >> (2 * v_im + 0)) & 0x01010101u) << 4;
        q0hi  |= ((qh >> (2 * v_im + 1)) & 0x01010101u) << 4;
        q64lo |= ((qh >> (2 * v_im + 4)) & 0x01010101u) << 4;
        q64hi |= ((qh >> (2 * v_im + 5)) & 0x01010101u) << 4;
    }
    const float sx = dot4(by10,  unpack_u8x4(q0lo));
    const float sy = dot4(by132, unpack_u8x4(q0hi));
    const float sz = dot4(by20,  unpack_u8x4(q64lo));
    const float sw = dot4(by232, unpack_u8x4(q64hi));
    const float smin = sum4(by10) * sc_lo.z + sum4(by132) * sc_lo.w + sum4(by20) * sc_hi.z + sum4(by232) * sc_hi.w;
    return fmaf(dm.x, fmaf(sx, sc_lo.x, fmaf(sy, sc_lo.y, fmaf(sz, sc_hi.x, sw * sc_hi.y))), -dm.y * smin);
}

// q6_K: 16 threads per 256-value superblock as in mul_mat_vec_q6_k.comp. Each thread owns four groups of
// four values at l0, l0+32, l0+64, l0+96 within its 128-value half; six-bit values are the low nibble of
// ql plus two bits of qh, minus 32; eight-bit signed scales, one per 16 values.
static __device__ __forceinline__ float q6_K_superblock_dot(
        const block_q6_K * b, const int v_im, const int l0,
        const float4 by0, const float4 by32, const float4 by64, const float4 by96) {
    const int ql_offset = 64 * v_im + l0;
    const int qh_offset = 32 * v_im + l0;
    const int s_offset  =  8 * v_im + l0 / 16;
    const uint32_t ql0  = *(const uint32_t *) (b->ql + ql_offset);
    const uint32_t ql32 = *(const uint32_t *) (b->ql + ql_offset + 32);
    const uint32_t qh   = *(const uint32_t *) (b->qh + qh_offset);
    const uint32_t q0 = ( ql0        & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4);
    const uint32_t q1 = ( ql32       & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2);
    const uint32_t q2 = ((ql0  >> 4) & 0x0F0F0F0Fu) |  (qh & 0x30303030u);
    const uint32_t q3 = ((ql32 >> 4) & 0x0F0F0F0Fu) | ((qh & 0xC0C0C0C0u) >> 2);
    // (q - 32) * y summed: subtract 32 * sum(y) once per group instead of per value
    const float s0 = dot4(by0,  unpack_u8x4(q0)) - 32.0f * sum4(by0);
    const float s1 = dot4(by32, unpack_u8x4(q1)) - 32.0f * sum4(by32);
    const float s2 = dot4(by64, unpack_u8x4(q2)) - 32.0f * sum4(by64);
    const float s3 = dot4(by96, unpack_u8x4(q3)) - 32.0f * sum4(by96);
    const float d = __half2float(b->d);
    return d * fmaf(s0, (float) b->scales[s_offset], fmaf(s1, (float) b->scales[s_offset + 2],
                    fmaf(s2, (float) b->scales[s_offset + 4], s3 * (float) b->scales[s_offset + 6])));
}

// q8_0: 32-value blocks; a thread takes eight consecutive values, four threads share a block, a wave
// covers eight blocks (256 values) per iteration, matching the superblock stride of the other types.
static __device__ __forceinline__ float q8_0_span_dot(
        const block_q8_0 * blk, const int sub, const float4 by0, const float4 by4) {
    const int8_t * q = blk->qs + 8 * sub;
    const float s = fmaf(by0.x, (float) q[0], fmaf(by0.y, (float) q[1], fmaf(by0.z, (float) q[2], fmaf(by0.w, (float) q[3],
                    fmaf(by4.x, (float) q[4], fmaf(by4.y, (float) q[5], fmaf(by4.z, (float) q[6], by4.w * (float) q[7])))))));
    return __half2float(blk->d) * s;
}

// ---------------------------------------------------------------------------------------------------
// The codebook in shared memory.
//
// A codebook lookup is a second dependent memory round trip, on the same path the weight stream uses,
// and there are four of them per thread per 32-value group. ggml-vulkan's IQ matvec shaders avoid that
// by calling init_iq_shmem before their loop, which stages the codebook in shared memory so every
// lookup is an LDS read. This does the same. A 32-lane block copies 512 dwords in sixteen loads per
// thread and then performs about eighty lookups per thread, so the copy pays for itself many times
// over -- but only where a block covers many superblocks, which the expert-id path never does, so the
// caller stages only for ordinary matrix-vector products. See logs/rdna1-iq-matvec-2026-09-20.
template <ggml_type type>
static constexpr int iq_grid_dwords() {
    return type == GGML_TYPE_IQ3_XXS ? 256 : 512;    // iq2_xxs is 256 entries of eight bytes
}

template <ggml_type type>
static __device__ __forceinline__ void iq_grid_stage(uint32_t * s_grid) {
    const uint32_t * src;
    if constexpr (type == GGML_TYPE_IQ2_XXS) {
        src = (const uint32_t *) iq2xxs_grid;
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        src = iq3xxs_grid;
    } else {
        src = iq3s_grid;
    }
    for (int i = threadIdx.x; i < iq_grid_dwords<type>(); i += 32) {
        s_grid[i] = src[i];
    }
    __syncthreads();
}

// One four-byte codebook entry, from shared memory when the block staged it there.
template <ggml_type type, bool lds>
static __device__ __forceinline__ uint32_t iq_grid32(const uint32_t * s_grid, const uint32_t idx) {
    if constexpr (lds) {
        return s_grid[idx];
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        return iq3xxs_grid[idx];
    } else {
        return iq3s_grid[idx];
    }
}

// iq2_xxs entries are eight bytes; returns the half selected by `hi`.
template <bool lds>
static __device__ __forceinline__ uint32_t iq_grid64(const uint32_t * s_grid, const uint32_t idx, const int hi) {
    if constexpr (lds) {
        return s_grid[2 * idx + hi];
    }
    return (uint32_t) (iq2xxs_grid[idx] >> (32 * hi));
}

// The IQ types: thread itid owns the 16 consecutive values 16*itid .. 16*itid+15 of the superblock, which
// is the second half (half = 1) or the first half of 32-value group ib32 = itid/2. Codebook grids and the
// sign tables come from ggml-common.h, the same ones vecdotq.cuh uses.
template <ggml_type type, bool lds>
static __device__ __forceinline__ float iq_superblock_dot(
        const void * blk, const int itid, const uint32_t * s_grid,
        const float4 by0, const float4 by4, const float4 by8, const float4 by12) {
    const int ib32 = itid / 2, half = itid & 1;
    if constexpr (type == GGML_TYPE_IQ2_XXS) {
        const block_iq2_xxs * b = (const block_iq2_xxs *) blk;
        const uint16_t * q2 = b->qs + 4 * ib32;
        const uint8_t  * aux8 = (const uint8_t *) q2;
        const uint32_t aux32 = q2[2] | (q2[3] << 16);
        const float db = __half2float(b->d) * (0.5f + (aux32 >> 28)) * 0.25f;
        const int l0 = 2 * half, l1 = l0 + 1;
        const uint32_t i0 = aux8[l0], i1 = aux8[l1];
        const uint32_t s0 = ksigns_iq2xs[(aux32 >> (7 * l0)) & 127], s1 = ksigns_iq2xs[(aux32 >> (7 * l1)) & 127];
        const float sum = dot4(by0,  signed4(unpack_u8x4(iq_grid64<lds>(s_grid, i0, 0)),         s0, 0))
                        + dot4(by4,  signed4(unpack_u8x4(iq_grid64<lds>(s_grid, i0, 1)), s0, 4))
                        + dot4(by8,  signed4(unpack_u8x4(iq_grid64<lds>(s_grid, i1, 0)),         s1, 0))
                        + dot4(by12, signed4(unpack_u8x4(iq_grid64<lds>(s_grid, i1, 1)), s1, 4));
        return db * sum;
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        const block_iq3_xxs * b = (const block_iq3_xxs *) blk;
        // the four codebook indices are the consecutive bytes 4*half..4*half+3 of the group, at an even
        // offset in every block, so two 16-bit loads replace four 8-bit ones
        const uint8_t  * q3  = b->qs + 8 * ib32 + 4 * half;
        const uint16_t * gas = (const uint16_t *) (b->qs + QK_K / 4) + 2 * ib32;
        const uint32_t aux32 = gas[0] | (gas[1] << 16);
        const float db = __half2float(b->d) * (0.5f + (aux32 >> 28)) * 0.5f;
        const int l0 = 2 * half;
        const uint32_t s0 = ksigns_iq2xs[(aux32 >> (7 * l0)) & 127];
        const uint32_t s1 = ksigns_iq2xs[(aux32 >> (7 * (l0 + 1))) & 127];
        const uint32_t q01 = *(const uint16_t *) (q3 + 0);
        const uint32_t q23 = *(const uint16_t *) (q3 + 2);
        const float sum = dot4(by0,  signed4(unpack_u8x4(iq_grid32<type, lds>(s_grid,  q01       & 0xFF)), s0, 0))
                        + dot4(by4,  signed4(unpack_u8x4(iq_grid32<type, lds>(s_grid, (q01 >> 8) & 0xFF)), s0, 4))
                        + dot4(by8,  signed4(unpack_u8x4(iq_grid32<type, lds>(s_grid,  q23       & 0xFF)), s1, 0))
                        + dot4(by12, signed4(unpack_u8x4(iq_grid32<type, lds>(s_grid, (q23 >> 8) & 0xFF)), s1, 4));
        return db * sum;
    } else {  // IQ3_S
        const block_iq3_s * b = (const block_iq3_s *) blk;
        // as above, and the two sign bytes are consecutive too: three 16-bit loads for six 8-bit ones
        const uint8_t * qs = b->qs + 8 * ib32 + 4 * half;
        const uint32_t qh = b->qh[ib32];
        const float db = __half2float(b->d) * (float) (1 + 2 * ((b->scales[ib32 / 2] >> (4 * (ib32 & 1))) & 0xf));
        const int l0 = 2 * half;
        const uint32_t sg = *(const uint16_t *) (b->signs + 4 * ib32 + l0);
        const uint32_t s0 = sg & 0xFF, s1 = sg >> 8;
        const uint32_t q01 = *(const uint16_t *) (qs + 0);
        const uint32_t q23 = *(const uint16_t *) (qs + 2);
        const int h0 = 8 - 2 * l0;      // the ninth index bit of each of the four entries
        const float sum = dot4(by0,  signed4(unpack_u8x4(iq_grid32<type, lds>(s_grid, ( q01       & 0xFF) | ((qh << (h0    )) & 256))), s0, 0))
                        + dot4(by4,  signed4(unpack_u8x4(iq_grid32<type, lds>(s_grid, ((q01 >> 8) & 0xFF) | ((qh << (h0 - 1)) & 256))), s0, 4))
                        + dot4(by8,  signed4(unpack_u8x4(iq_grid32<type, lds>(s_grid, ( q23       & 0xFF) | ((qh << (h0 - 2)) & 256))), s1, 0))
                        + dot4(by12, signed4(unpack_u8x4(iq_grid32<type, lds>(s_grid, ((q23 >> 8) & 0xFF) | ((qh << (h0 - 3)) & 256))), s1, 4));
        return db * sum;
    }
}

template <ggml_type type, bool has_fusion, bool lds>
static __global__ void __launch_bounds__(32) mul_mat_vec_f32_rdna1(
        const void * __restrict__ vx, const float * __restrict__ y, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int64_t stride_row_x,
        const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst, const int nchannels_y,
        const void * __restrict__ vgate, const float * __restrict__ x_bias, const float * __restrict__ gate_bias,
        const int glu_op) {
    const int channel = blockIdx.y;                             // dst channel: expert slot when ids, else batch channel
    const int channel_x = ids ? ids[channel] : channel;         // the weight matrix it reads
    const char * x = (const char *) vx + channel_x * stride_channel_x;
    const char * g = has_fusion && vgate ? (const char *) vgate + channel_x * stride_channel_x : nullptr;
    y   += (channel % nchannels_y) * stride_channel_y;     // MUL_MAT_ID's gate/up broadcast one vector to every expert
    dst += channel * stride_channel_dst;

    const int row0 = MMVQ_F32_ROWS * blockIdx.x;
    const int nspans = ncols_x / 256;                // 256 values per iteration for every type here
    const int tid  = threadIdx.x;                   // 0..31

    float temp[MMVQ_F32_ROWS] = {0.0f};
    float temp_gate[MMVQ_F32_ROWS] = {0.0f};

    if constexpr (type == GGML_TYPE_Q8_0) {
        const int blk = tid / 4, sub = tid % 4;      // block within the span, eight-value slice within it
        for (int i = 0; i < nspans; ++i) {
            const float4 by0 = ld4(y + i * 256 + blk * 32 + sub * 8);
            const float4 by4 = ld4(y + i * 256 + blk * 32 + sub * 8 + 4);
#pragma unroll
            for (int n = 0; n < MMVQ_F32_ROWS; ++n) {
                const int row = row0 + n;
                if (row >= nrows_x) break;
                const block_q8_0 * bx = (const block_q8_0 *) (x + row * stride_row_x) + i * 8 + blk;
                temp[n] += q8_0_span_dot(bx, sub, by0, by4);
                if constexpr (has_fusion) {
                    if (g) temp_gate[n] += q8_0_span_dot((const block_q8_0 *) (g + row * stride_row_x) + i * 8 + blk, sub, by0, by4);
                }
            }
        }
    } else {
        const int itid = tid % 16;                  // position inside a superblock
        const int ix   = tid / 16;                  // which superblock of the pair
        if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            using block_t = typename std::conditional<type == GGML_TYPE_Q4_K, block_q4_K, block_q5_K>::type;
            constexpr bool has_qh = type == GGML_TYPE_Q5_K;
            const int il = itid / 4, ir = itid % 4, v_im = il / 2, v_in = il % 2;
            const int l0 = 4 * (2 * ir + v_in);
            const int q_offset = 32 * v_im + l0;
            const int y_offset = 64 * v_im + l0;
            for (int i = ix; i < nspans; i += 2) {
                const float4 by10  = ld4(y + i * QK_K + y_offset);
                const float4 by132 = ld4(y + i * QK_K + y_offset + 32);
                const float4 by20  = ld4(y + i * QK_K + y_offset + 128);
                const float4 by232 = ld4(y + i * QK_K + y_offset + 160);
#pragma unroll
                for (int n = 0; n < MMVQ_F32_ROWS; ++n) {
                    const int row = row0 + n;
                    if (row >= nrows_x) break;
                    temp[n] += q45_K_superblock_dot<block_t, has_qh>((const block_t *) (x + row * stride_row_x) + i, v_im, q_offset, by10, by132, by20, by232);
                    if constexpr (has_fusion) {
                        if (g) temp_gate[n] += q45_K_superblock_dot<block_t, has_qh>((const block_t *) (g + row * stride_row_x) + i, v_im, q_offset, by10, by132, by20, by232);
                    }
                }
            }
        } else if constexpr (type == GGML_TYPE_Q6_K) {
            const int v_im = itid / 8, v_in = itid % 8;
            const int l0 = 4 * v_in;
            const int y_offset = 128 * v_im + l0;
            for (int i = ix; i < nspans; i += 2) {
                const float4 by0  = ld4(y + i * QK_K + y_offset);
                const float4 by32 = ld4(y + i * QK_K + y_offset + 32);
                const float4 by64 = ld4(y + i * QK_K + y_offset + 64);
                const float4 by96 = ld4(y + i * QK_K + y_offset + 96);
#pragma unroll
                for (int n = 0; n < MMVQ_F32_ROWS; ++n) {
                    const int row = row0 + n;
                    if (row >= nrows_x) break;
                    temp[n] += q6_K_superblock_dot((const block_q6_K *) (x + row * stride_row_x) + i, v_im, l0, by0, by32, by64, by96);
                    if constexpr (has_fusion) {
                        if (g) temp_gate[n] += q6_K_superblock_dot((const block_q6_K *) (g + row * stride_row_x) + i, v_im, l0, by0, by32, by64, by96);
                    }
                }
            }
        } else {  // IQ2_XXS, IQ3_XXS, IQ3_S: 16 consecutive values per thread
            __shared__ uint32_t s_grid[lds ? iq_grid_dwords<type>() : 1];
            if constexpr (lds) {
                iq_grid_stage<type>(s_grid);
            }
            constexpr size_t bsz = type == GGML_TYPE_IQ2_XXS ? sizeof(block_iq2_xxs) : type == GGML_TYPE_IQ3_XXS ? sizeof(block_iq3_xxs)
                                 : sizeof(block_iq3_s);
            const int y_offset = 16 * itid;
            for (int i = ix; i < nspans; i += 2) {
                const float4 by0  = ld4(y + i * QK_K + y_offset);
                const float4 by4  = ld4(y + i * QK_K + y_offset + 4);
                const float4 by8  = ld4(y + i * QK_K + y_offset + 8);
                const float4 by12 = ld4(y + i * QK_K + y_offset + 12);
#pragma unroll
                for (int n = 0; n < MMVQ_F32_ROWS; ++n) {
                    const int row = row0 + n;
                    if (row >= nrows_x) break;
                    const size_t off = (size_t) i * bsz;
                    temp[n] += iq_superblock_dot<type, lds>(x + row * stride_row_x + off, itid, s_grid, by0, by4, by8, by12);
                    if constexpr (has_fusion) {
                        if (g) temp_gate[n] += iq_superblock_dot<type, lds>(g + row * stride_row_x + off, itid, s_grid, by0, by4, by8, by12);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int n = 0; n < MMVQ_F32_ROWS; ++n) {
        temp[n] = warp_reduce_sum<32>(temp[n]);
        if constexpr (has_fusion) temp_gate[n] = warp_reduce_sum<32>(temp_gate[n]);
    }
    if (tid < MMVQ_F32_ROWS && row0 + tid < nrows_x) {
        float result = temp[tid];
        if constexpr (has_fusion) {
            const int row = row0 + tid;
            const int channel_bias = ids ? channel_x : channel;     // as mul_mat_vec_q: per-expert biases with ids
            if (x_bias) result += x_bias[channel_bias * stride_channel_dst + row];
            if (g) {
                float gate_value = temp_gate[tid];
                if (gate_bias) gate_value += gate_bias[channel_bias * stride_channel_dst + row];
                switch (glu_op) {
                    case GGML_GLU_OP_SWIGLU:     result *= ggml_cuda_op_silu_single(gate_value); break;
                    case GGML_GLU_OP_GEGLU:      result *= ggml_cuda_op_gelu_single(gate_value); break;
                    case GGML_GLU_OP_SWIGLU_OAI: result  = ggml_cuda_op_swiglu_oai_single(gate_value, result); break;
                    default:                     result  = result * gate_value; break;
                }
            }
        }
        dst[row0 + tid] = result;
    }
}

bool ggml_cuda_mmvq_rdna1_f32_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                        const ggml_tensor * dst, const int cc) {
    switch (src0->type) {
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S:
            break;
        default:
            return false;
    }
    if (!GGML_CUDA_CC_IS_RDNA1(cc) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src0->ne[0] % 256 != 0 || src0->ne[3] != 1 || src1->ne[3] != 1 || !ggml_is_contiguous(src1) ||
        src0->nb[1] % ggml_type_size(src0->type) != 0) {
        return false;
    }
    if (ids) {
        // MUL_MAT_ID with one token: src1 [K, n_used, 1], ids [n_used, 1], dst [nrows, n_used, 1]
        return src1->ne[2] == 1 && ids->ne[1] == 1 && ids->ne[0] == dst->ne[1] && dst->ne[2] == 1 &&
               (src1->ne[1] == 1 || src1->ne[1] == dst->ne[1]) && dst->ne[1] <= 65535;
    }
    return src1->ne[1] == 1 && src1->ne[2] == src0->ne[2];
}

template <ggml_type type, bool lds>
static void launch_f32_rdna1(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                             const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_device * fusion) {
    const int ncols_x = src0->ne[0];
    const int nrows_x = src0->ne[1];
    const int nchannels   = ids ? dst->ne[1]  : src0->ne[2];
    const int nchannels_y = ids ? src1->ne[1] : src1->ne[2];
    const dim3 grid((nrows_x + MMVQ_F32_ROWS - 1) / MMVQ_F32_ROWS, nchannels, 1);
    // strides in bytes for the weights (row stride is a whole number of blocks), in floats for y and dst;
    // with ids the channel of y and dst is dimension 1 (MUL_MAT_ID layout), otherwise dimension 2
    const int64_t srx = src0->nb[1], scx = src0->nb[2];
    const int64_t scy = (ids ? src1->nb[1] : src1->nb[2]) / sizeof(float);
    const int64_t scd = (ids ? dst->nb[1]  : dst->nb[2])  / sizeof(float);
    const int32_t * ids_d = ids ? (const int32_t *) ids->data : nullptr;
    const bool fused = fusion && (fusion->x_bias || fusion->gate);
    if (fused) {
        mul_mat_vec_f32_rdna1<type, true, lds><<<grid, 32, 0, ctx.stream()>>>(
            src0->data, (const float *) src1->data, ids_d, (float *) dst->data, ncols_x, nrows_x, srx, scx, scy, scd, nchannels_y,
            fusion->gate, (const float *) fusion->x_bias, (const float *) fusion->gate_bias, (int) fusion->glu_op);
    } else {
        mul_mat_vec_f32_rdna1<type, false, lds><<<grid, 32, 0, ctx.stream()>>>(
            src0->data, (const float *) src1->data, ids_d, (float *) dst->data, ncols_x, nrows_x, srx, scx, scy, scd, nchannels_y,
            nullptr, nullptr, nullptr, 0);
    }
}

void ggml_cuda_mmvq_rdna1_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                              const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_device * fusion) {
    // staging the codebook costs one copy per block, which only pays where a block covers many
    // superblocks; an expert-id launch covers a handful, so those keep the global lookup
    const bool stage = ids == nullptr;
    switch (src0->type) {
        case GGML_TYPE_Q4_K:    launch_f32_rdna1<GGML_TYPE_Q4_K, false>(ctx, src0, src1, ids, dst, fusion); break;
        case GGML_TYPE_Q5_K:    launch_f32_rdna1<GGML_TYPE_Q5_K, false>(ctx, src0, src1, ids, dst, fusion); break;
        case GGML_TYPE_Q6_K:    launch_f32_rdna1<GGML_TYPE_Q6_K, false>(ctx, src0, src1, ids, dst, fusion); break;
        case GGML_TYPE_Q8_0:    launch_f32_rdna1<GGML_TYPE_Q8_0, false>(ctx, src0, src1, ids, dst, fusion); break;
        case GGML_TYPE_IQ2_XXS:
            if (stage) launch_f32_rdna1<GGML_TYPE_IQ2_XXS, true >(ctx, src0, src1, ids, dst, fusion);
            else       launch_f32_rdna1<GGML_TYPE_IQ2_XXS, false>(ctx, src0, src1, ids, dst, fusion);
            break;
        case GGML_TYPE_IQ3_XXS:
            if (stage) launch_f32_rdna1<GGML_TYPE_IQ3_XXS, true >(ctx, src0, src1, ids, dst, fusion);
            else       launch_f32_rdna1<GGML_TYPE_IQ3_XXS, false>(ctx, src0, src1, ids, dst, fusion);
            break;
        case GGML_TYPE_IQ3_S:
            if (stage) launch_f32_rdna1<GGML_TYPE_IQ3_S, true >(ctx, src0, src1, ids, dst, fusion);
            else       launch_f32_rdna1<GGML_TYPE_IQ3_S, false>(ctx, src0, src1, ids, dst, fusion);
            break;
        default: GGML_ABORT("unsupported type");
    }
}
