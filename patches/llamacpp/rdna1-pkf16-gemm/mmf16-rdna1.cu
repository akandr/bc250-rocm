// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// A packed-fp16 tile GEMM for RDNA1 prefill, dequantising inside the kernel.
//
// gfx1013 has no v_dot4_i32_i8, so MMQ's int8 dot products are emulated in four multiplies and four
// adds and run at 2.08 Tmac/s, which is where ROCm's prefill sits. v_pk_fma_f16 on the same chip runs
// at 14.4 TFLOP/s (logs/alu-rates-2026-09-19). This multiplies in that format.
//
// The first version of this kernel took its weights already in f16, which meant materialising the whole
// weight matrix before every matmul: correct and faster per kernel, but slower per model, because a
// [12288,4096] q8_0 weight becomes 100 MB written and read back where MMQ reads 53 MB once
// (logs/round6-2026-09-19). So the tiles are dequantised into shared memory here, one BM x BK block at a
// time, exactly as MMQ does with int8 and ggml-vulkan's mul_mm.comp does in f16, and the activations are
// read as f32 and converted while staging. Nothing is materialised.
//
// Accumulation is f16 within a BK step and promoted to f32 at every step: without the promotion the
// error over K = 8960 is 1.4 percent rms, with it 0.088, below the quantisation error.
//
// Layout, as cuBLAS is called with: A row m contiguous in k, B column n contiguous in k, C column-major.
#include "mmf16-rdna1.cuh"
#include "mmid.cuh"
#include "convert.cuh"
#include "dequantize.cuh"
#include <cstdlib>

// The tile. BN is the column (token) tile and it is 64, not the 128 a standalone prototype preferred:
// in situ 128 leaves only four column tiles at a 512-token batch and takes 221 registers, four waves a
// SIMD, where 64 takes 125 and gets eight. 64 is ahead at every batch from 256 to 2048 on every model
// measured, by 1.07 to 1.55 times (logs/rdna1-pkf16-tile-2026-09-20).
#define MMF16_BM 128
#define MMF16_BN  64
#define MMF16_BK  32
#define MMF16_TM   8
#define MMF16_TN   8
#define MMF16_NT  ((MMF16_BM / MMF16_TM) * (MMF16_BN / MMF16_TN))    // 128 threads, one per A-tile row


// Dequantise rows [row0, row0+BM) of a quantised matrix, elements [k0, k0+BK), into As[k][m].
// One thread per row; BK is 32, so every row's 32 elements live in one 32-value group of one superblock.
template <ggml_type type, int BM>
static __device__ __forceinline__ void mmf16_load_a(
        const char * __restrict__ A, half (&As)[MMF16_BK][BM + 8],
        const int row0, const int k0, const int M, const size_t row_size, const int tid) {
    const int m = tid;                                   // NT == BM, one row each
    const int row = row0 + m;
    if (row >= M) {
#pragma unroll
        for (int k = 0; k < MMF16_BK; ++k) As[k][m] = __float2half(0.0f);
        return;
    }
    const char * rp = A + (size_t) row * row_size;

    if constexpr (type == GGML_TYPE_Q4_K) {
        const block_q4_K * b = (const block_q4_K *) rp + k0 / QK_K;
        const int j0 = k0 % QK_K;                        // 0, 32, 64 ... 224
        const int is = j0 / 32;
        const float dall = __low2float(b->dm), dmin = __high2float(b->dm);
        uint8_t sc, mb;
        get_scale_min_k4(is, b->scales, sc, mb);
        const float d = dall * sc, mn = dmin * mb;
        const uint8_t * q = b->qs + 32 * (is / 2);
        const bool high = is % 2;
#pragma unroll
        for (int k = 0; k < MMF16_BK; ++k) {
            const int v = high ? (q[k] >> 4) : (q[k] & 0xF);
            As[k][m] = __float2half(d * v - mn);
        }
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        const block_q6_K * b = (const block_q6_K *) rp + k0 / QK_K;
        const int j0 = k0 % QK_K;                        // 0, 32, ... 224: the 32-value group
        const float d = __half2float(b->d);
        // ggml's layout: value j = (ql[64*(j/128) + (j%64)] >> (4*((j%128)/64)) & 0xF)
        //                       | ((qh[32*(j/128) + (j%32)] >> (2*((j%128)/32))) & 3) << 4, minus 32
        const int hi   = j0 / 128;                       // which 128-value half
        const int in   = j0 % 128;                       // 0, 32, 64 or 96 inside it
        const uint8_t * ql = b->ql + 64 * hi + (in % 64);
        const uint8_t * qh = b->qh + 32 * hi;
        const int lshift = 4 * (in / 64);                // low or high nibble of ql
        const int hshift = 2 * (in / 32);                // which bit pair of qh
#pragma unroll
        for (int k = 0; k < MMF16_BK; ++k) {
            const int q = (int) ((ql[k] >> lshift) & 0xF) | (int) (((qh[(in % 32) + k] >> hshift) & 3) << 4);
            As[k][m] = __float2half(d * (float) b->scales[(j0 + k) / 16] * (float) (q - 32));
        }
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        const block_q5_K * b = (const block_q5_K *) rp + k0 / QK_K;
        const int j0 = k0 % QK_K, is = j0 / 32;
        const float dall = __low2float(b->dm), dmin = __high2float(b->dm);
        uint8_t sc, mb;
        get_scale_min_k4(is, b->scales, sc, mb);
        const float d = dall * sc, mn = dmin * mb;
        const uint8_t * q = b->qs + 32 * (is / 2);
        const uint8_t * qh = b->qh;
        const bool high = is % 2;
        const uint8_t hm = 1u << is;
#pragma unroll
        for (int k = 0; k < MMF16_BK; ++k) {
            const int v = (high ? (q[k] >> 4) : (q[k] & 0xF)) | ((qh[k] & hm) ? 16 : 0);
            As[k][m] = __float2half(d * v - mn);
        }
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        const block_iq2_xxs * b = (const block_iq2_xxs *) rp + k0 / QK_K;
        const int ib32 = (k0 % QK_K) / 32;
        const uint16_t * q2 = b->qs + 4 * ib32;
        const uint8_t * aux8 = (const uint8_t *) q2;
        const uint32_t aux32 = q2[2] | ((uint32_t) q2[3] << 16);
        const float db = __half2float(b->d) * (0.5f + (aux32 >> 28)) * 0.25f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint8_t * grid = (const uint8_t *) (iq2xxs_grid + aux8[l]);
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
#pragma unroll
            for (int j = 0; j < 8; ++j) As[8 * l + j][m] = __float2half(db * grid[j] * (signs & kmask_iq2xs[j] ? -1.0f : 1.0f));
        }
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        const block_iq3_xxs * b = (const block_iq3_xxs *) rp + k0 / QK_K;
        const int ib32 = (k0 % QK_K) / 32;
        const uint8_t * q3 = b->qs + 8 * ib32;
        const uint16_t * gas = (const uint16_t *) (b->qs + QK_K / 4) + 2 * ib32;
        const uint32_t aux32 = gas[0] | ((uint32_t) gas[1] << 16);
        const float db = __half2float(b->d) * (0.5f + (aux32 >> 28)) * 0.5f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint8_t * g1 = (const uint8_t *) (iq3xxs_grid + q3[2 * l + 0]);
            const uint8_t * g2 = (const uint8_t *) (iq3xxs_grid + q3[2 * l + 1]);
            const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                As[8 * l + j][m]     = __float2half(db * g1[j] * (signs & kmask_iq2xs[j]     ? -1.0f : 1.0f));
                As[8 * l + j + 4][m] = __float2half(db * g2[j] * (signs & kmask_iq2xs[j + 4] ? -1.0f : 1.0f));
            }
        }
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        const block_iq3_s * b = (const block_iq3_s *) rp + k0 / QK_K;
        const int ib32 = (k0 % QK_K) / 32;
        const uint8_t * qs = b->qs + 8 * ib32;
        const uint32_t qh = b->qh[ib32];
        const float db = __half2float(b->d) * (1 + 2 * ((b->scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf));
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint8_t * g1 = (const uint8_t *) (iq3s_grid + (qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)));
            const uint8_t * g2 = (const uint8_t *) (iq3s_grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
            const uint8_t signs = b->signs[4 * ib32 + l];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                As[8 * l + j][m]     = __float2half(db * g1[j] * (signs & kmask_iq2xs[j]     ? -1.0f : 1.0f));
                As[8 * l + j + 4][m] = __float2half(db * g2[j] * (signs & kmask_iq2xs[j + 4] ? -1.0f : 1.0f));
            }
        }
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const block_iq4_xs * b = (const block_iq4_xs *) rp + k0 / QK_K;
        const int ib32 = (k0 % QK_K) / 32;
        const int ls = ((b->scales_l[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf) | (((b->scales_h >> (2 * ib32)) & 3) << 4);
        const float db = __half2float(b->d) * (ls - 32);
        const uint8_t * q4 = b->qs + 16 * ib32;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            As[j][m]      = __float2half(db * kvalues_iq4nl[q4[j] & 0xF]);
            As[j + 16][m] = __float2half(db * kvalues_iq4nl[q4[j] >> 4]);
        }
    } else if constexpr (type == GGML_TYPE_Q8_0) {
        const block_q8_0 * b = (const block_q8_0 *) rp + k0 / QK8_0;
#pragma unroll
        for (int k = 0; k < MMF16_BK; ++k) {
            const block_q8_0 * bb = b + k / QK8_0;
            As[k][m] = __float2half(__half2float(bb->d) * (float) bb->qs[k % QK8_0]);
        }
    }
}

// The tile is a template parameter because the shipped 128x128 leaves only four column tiles at a
// 512-token batch, which is the batch the benchmarks use; see logs/rdna1-pkf16-tile-2026-09-20.
// mmf16_load_a puts one thread on each row of the A tile, so the block must have exactly BM threads,
// which constrains BN / TN == TM.
// The tile is a template parameter because the shipped 128x128 leaves only four column tiles at a
// 512-token batch, which is the batch the benchmarks use; see logs/rdna1-pkf16-tile-2026-09-20.
// mmf16_load_a puts one thread on each row of the A tile, so the block must have exactly BM threads,
// which constrains BN / TN == TM.
//
// mmid selects the expert path. There the columns are the compact, expert-sorted slots that
// ggml_cuda_launch_mm_ids_helper builds: block z takes one expert, its column range comes from
// expert_bounds, the activation column of compact slot c is ids_src1[c] and its destination column is
// ids_dst[c]. A block walks its expert's column tiles in a loop, so the grid does not have to be sized
// for the most heavily used expert. With mmid false the loop runs exactly once and the kernel is what
// it was.
template <ggml_type type, int promote, bool dbuf, bool mmid = false, int BM = MMF16_BM, int BN = MMF16_BN,
          int TM = MMF16_TM, int TN = MMF16_TN>   // promote: stages between f32 promotions; dbuf: prefetch the next tile
static __global__ void __launch_bounds__(BM) mmf16_rdna1_gemm(
        const char * __restrict__ A, const float * __restrict__ B, float * __restrict__ C,
        const int M, const int N, const int K, const size_t row_size, const int sb, const int ldc,
        const int32_t * __restrict__ ids_src1 = nullptr, const int32_t * __restrict__ ids_dst = nullptr,
        const int32_t * __restrict__ expert_bounds = nullptr, const size_t stride_expert = 0) {
    // the A-tile loader is one thread per row, so the block has exactly BM threads
    static_assert((BM / TM) * (BN / TN) == BM, "bad tile: (BM/TM)*(BN/TN) must be BM, so BN/TN == TM");
    static_assert(TN % 2 == 0, "the B tile is read as half2 pairs");
    __shared__ half As[dbuf ? 2 : 1][MMF16_BK][BM + 8];
    __shared__ half Bs[dbuf ? 2 : 1][MMF16_BK][BN + 8];

    const int tid  = threadIdx.x;
    const int tm   = (tid % (BM / TM)) * TM;
    const int tn   = (tid / (BM / TM)) * TN;
    const int row0 = blockIdx.x * BM;

    if constexpr (mmid) {
        A += (size_t) blockIdx.z * stride_expert;
    }
    const int cbeg = mmid ? expert_bounds[blockIdx.z]     : 0;
    const int cend = mmid ? expert_bounds[blockIdx.z + 1] : N;

    for (int col0 = cbeg + blockIdx.y * BN; col0 < cend; col0 += gridDim.y * BN) {
        // the activation column a compact slot reads from, or the slot itself off the expert path
        auto bcol = [&](const int c) { return mmid ? ids_src1[c] : c; };

        half2 acc[TN][TM / 2];
        float accf[TN][TM];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
#pragma unroll
            for (int i = 0; i < TM / 2; ++i) acc[j][i] = make_half2(0.0f, 0.0f);
#pragma unroll
            for (int i = 0; i < TM; ++i) accf[j][i] = 0.0f;
        }

        int stage = 0;
        if (dbuf) {                                     // prime the first tile
            mmf16_load_a<type, BM>(A, As[0], row0, 0, M, row_size, tid);
#pragma unroll
            for (int idx = tid; idx < BN * MMF16_BK; idx += BM) {
                const int n = idx / MMF16_BK, k = idx % MMF16_BK;
                Bs[0][k][n] = col0 + n < cend ? __float2half(B[(size_t) bcol(col0 + n) * sb + k]) : __float2half(0.0f);
            }
        }
        for (int k0 = 0; k0 < K; k0 += MMF16_BK, ++stage) {
            const int cur = dbuf ? (stage & 1) : 0;
            const int nxt = dbuf ? (cur ^ 1)   : 0;
            if (!dbuf) {
                mmf16_load_a<type, BM>(A, As[0], row0, k0, M, row_size, tid);
#pragma unroll
                for (int idx = tid; idx < BN * MMF16_BK; idx += BM) {
                    const int n = idx / MMF16_BK, k = idx % MMF16_BK;
                    Bs[0][k][n] = col0 + n < cend ? __float2half(B[(size_t) bcol(col0 + n) * sb + (k0 + k)]) : __float2half(0.0f);
                }
            }
            __syncthreads();

            if (dbuf && k0 + MMF16_BK < K) {            // fetch the next tile over this stage's arithmetic
                mmf16_load_a<type, BM>(A, As[nxt], row0, k0 + MMF16_BK, M, row_size, tid);
#pragma unroll
                for (int idx = tid; idx < BN * MMF16_BK; idx += BM) {
                    const int n = idx / MMF16_BK, k = idx % MMF16_BK;
                    Bs[nxt][k][n] = col0 + n < cend ? __float2half(B[(size_t) bcol(col0 + n) * sb + (k0 + MMF16_BK + k)]) : __float2half(0.0f);
                }
            }

#pragma unroll
            for (int k = 0; k < MMF16_BK; ++k) {
                half2 a[TM / 2], b[TN];
#pragma unroll
                for (int i = 0; i < TM / 2; ++i) a[i] = *(const half2 *) &As[cur][k][tm + 2 * i];
#pragma unroll
                for (int j = 0; j < TN; j += 2) {
                    const half2 pair = *(const half2 *) &Bs[cur][k][tn + j];
                    b[j]     = __low2half2(pair);
                    b[j + 1] = __high2half2(pair);
                }
#pragma unroll
                for (int j = 0; j < TN; ++j)
#pragma unroll
                    for (int i = 0; i < TM / 2; ++i) acc[j][i] = __hfma2(a[i], b[j], acc[j][i]);
            }
            if (promote == 1 || (stage % promote) == promote - 1 || k0 + MMF16_BK >= K) {
#pragma unroll
                for (int j = 0; j < TN; ++j)
#pragma unroll
                    for (int i = 0; i < TM / 2; ++i) {
                        accf[j][2 * i]     += __low2float(acc[j][i]);
                        accf[j][2 * i + 1] += __high2float(acc[j][i]);
                        acc[j][i] = make_half2(0.0f, 0.0f);
                    }
            }
            __syncthreads();
        }

#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = col0 + tn + j;
            if (c >= cend) continue;
            float * dst_col = C + (size_t) (mmid ? ids_dst[c] : c) * ldc;
#pragma unroll
            for (int i = 0; i < TM; i += 4) {
                const int m = row0 + tm + i;
                if (m + 3 < M) {
                    *(float4 *) &dst_col[m] = make_float4(accf[j][i], accf[j][i+1], accf[j][i+2], accf[j][i+3]);
                } else {
#pragma unroll
                    for (int t = 0; t < 4; ++t) if (m + t < M) dst_col[m + t] = accf[j][i + t];
                }
            }
        }
    }
}

static bool q8_enabled() {
    static const bool e = [] { const char * v = getenv("GGML_RDNA1_PKF16_Q8"); return v ? atoi(v) != 0 : true; }();
    return e;
}

// Below these the kernel is behind MMQ, because one tile of columns or rows does not fill the GPU, and
// both are switchable so they can be measured again on another part or after another tile change.
//
// The column threshold was 256 when the tile was 128 wide, where 128 tokens measured 22 percent behind
// MMQ. At a 64-wide tile 128 tokens is 1.21 to 1.24 times ahead on every model measured and 64 tokens is
// still behind, so the crossover is 128 (logs/rdna1-pkf16-thresholds-2026-09-21).
static int mmf16_min_cols() {
    static const int v = [] { const char * e = getenv("GGML_RDNA1_PKF16_MINCOLS"); return e ? atoi(e) : 128; }();
    return v;
}
static int mmf16_min_rows() {
    static const int v = [] { const char * e = getenv("GGML_RDNA1_PKF16_MINROWS"); return e ? atoi(e) : 512; }();
    return v;
}

// The types the kernel has a decoder for, and the master switch. Shared with the expert path, which
// applies its own shape tests.
static bool mmf16_type_supported(const ggml_tensor * src0, const int cc) {
    static const bool enabled = [] { const char * e = getenv("GGML_RDNA1_PKF16"); return e ? atoi(e) != 0 : true; }();
    if (!enabled || !GGML_CUDA_CC_IS_RDNA1(cc)) return false;
    switch (src0->type) {
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_XS:
            break;
        case GGML_TYPE_Q8_0:
            // q8_0 was left to MMQ when this kernel shipped, on a measurement that had it 15 to 20
            // percent behind (logs/round11-2026-09-20). That was at the 128x128 tile. At 128x64 it is
            // 1.44 to 1.76 times MMQ from 256 to 2048 tokens on the 8B, so it takes this path too;
            // GGML_RDNA1_PKF16_Q8=0 puts it back on MMQ (logs/rdna1-pkf16-tile-2026-09-20).
            if (!q8_enabled()) return false;
            break;
        default:
            return false;
    }
    return true;
}

bool ggml_cuda_mmf16_rdna1_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    if (!mmf16_type_supported(src0, cc)) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) return false;
    if (!ggml_is_contiguously_allocated(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) return false;
    if (src0->ne[0] % 256 != 0) return false;            // one 32-value group per stage, inside a superblock
    if (src1->ne[1] < mmf16_min_cols()) return false;
    if (src0->ne[1] < mmf16_min_rows()) return false;
    return true;
}

void ggml_cuda_mmf16_rdna1(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int M = src0->ne[1], N = src1->ne[1], K = src0->ne[0];
    const size_t row_size = ggml_row_size(src0->type, src0->ne[0]);
    cudaStream_t stream = ctx.stream();
    const char * A = (const char *) src0->data;
    const float * B = (const float *) src1->data;
    float * C = (float *) dst->data;

    // GGML_RDNA1_PKF16_TILE selects the tile at run time: 0 the shipped 128x128, then three shapes with
    // more column tiles for small batches. BN / TN == TM always, because the A-tile loader is one thread
    // per row and the block therefore has exactly BM threads.
    static const int tile = [] { const char * e = getenv("GGML_RDNA1_PKF16_TILE"); return e ? atoi(e) : 0; }();
    if (tile > 0) {   // 1 restores the 128x128 the kernel first shipped with; the rest are for retesting
#define MMF16_LAUNCH_TILE(t, bm, bn, tm, tn)                                                            \
        do {                                                                                            \
            const dim3 g((M + (bm) - 1) / (bm), (N + (bn) - 1) / (bn), 1);                              \
            mmf16_rdna1_gemm<t, 1, false, false, bm, bn, tm, tn><<<g, bm, 0, stream>>>(                        \
                A, B, C, M, N, K, row_size, K, dst->ne[0]);                                             \
        } while (0)
#define MMF16_TILE_SWITCH(t)                                                                            \
        do {                                                                                            \
            if      (tile == 1) MMF16_LAUNCH_TILE(t, 128, 128, 8, 16);                                  \
            else if (tile == 2) MMF16_LAUNCH_TILE(t, 128,  32, 8,  4);                                  \
            else if (tile == 3) MMF16_LAUNCH_TILE(t,  64,  64, 4, 16);                                  \
            else                MMF16_LAUNCH_TILE(t,  64, 128, 4, 32);                                  \
        } while (0)
        switch (src0->type) {
            case GGML_TYPE_Q4_K:    MMF16_TILE_SWITCH(GGML_TYPE_Q4_K);    break;
            case GGML_TYPE_Q5_K:    MMF16_TILE_SWITCH(GGML_TYPE_Q5_K);    break;
            case GGML_TYPE_IQ2_XXS: MMF16_TILE_SWITCH(GGML_TYPE_IQ2_XXS); break;
            case GGML_TYPE_IQ3_XXS: MMF16_TILE_SWITCH(GGML_TYPE_IQ3_XXS); break;
            case GGML_TYPE_IQ3_S:   MMF16_TILE_SWITCH(GGML_TYPE_IQ3_S);   break;
            case GGML_TYPE_IQ4_XS:  MMF16_TILE_SWITCH(GGML_TYPE_IQ4_XS);  break;
            case GGML_TYPE_Q6_K:    MMF16_TILE_SWITCH(GGML_TYPE_Q6_K);    break;
            case GGML_TYPE_Q8_0:    MMF16_TILE_SWITCH(GGML_TYPE_Q8_0);    break;
            default: GGML_ABORT("unsupported type");
        }
#undef MMF16_TILE_SWITCH
#undef MMF16_LAUNCH_TILE
        return;
    }

    const dim3 grid((M + MMF16_BM - 1) / MMF16_BM, (N + MMF16_BN - 1) / MMF16_BN, 1);
    switch (src0->type) {
    static const int promote = [] { const char * e = getenv("GGML_RDNA1_PKF16_PROMOTE"); const int v = e ? atoi(e) : 1;
                                    return (v == 1 || v == 2 || v == 4) ? v : 1; }();
    // prefetching the next tile doubles the shared memory and costs a wave of occupancy: measured no
    // better on the 1.5B and 27B and 19 percent worse on the 14B, at the 128x128 tile of the time
    static const bool dbuf = [] { const char * e = getenv("GGML_RDNA1_PKF16_DBUF"); return e ? atoi(e) != 0 : false; }();
#define MMF16_LAUNCH_PD(t, p, d) mmf16_rdna1_gemm<t, p, d><<<grid, MMF16_NT, 0, stream>>>(A, B, C, M, N, K, row_size, K, dst->ne[0])
#define MMF16_LAUNCH_P(t, p) do { if (dbuf) { MMF16_LAUNCH_PD(t, p, true); } else { MMF16_LAUNCH_PD(t, p, false); } } while (0)
#define MMF16_LAUNCH(t) do { if (promote == 2) { MMF16_LAUNCH_P(t, 2); } else if (promote == 4) { MMF16_LAUNCH_P(t, 4); } \
                             else { MMF16_LAUNCH_P(t, 1); } } while (0)
        case GGML_TYPE_Q4_K:    MMF16_LAUNCH(GGML_TYPE_Q4_K);    break;
        case GGML_TYPE_Q5_K:    MMF16_LAUNCH(GGML_TYPE_Q5_K);    break;
        case GGML_TYPE_IQ2_XXS: MMF16_LAUNCH(GGML_TYPE_IQ2_XXS); break;
        case GGML_TYPE_IQ3_XXS: MMF16_LAUNCH(GGML_TYPE_IQ3_XXS); break;
        case GGML_TYPE_IQ3_S:   MMF16_LAUNCH(GGML_TYPE_IQ3_S);   break;
        case GGML_TYPE_IQ4_XS:  MMF16_LAUNCH(GGML_TYPE_IQ4_XS);  break;
        case GGML_TYPE_Q6_K:    MMF16_LAUNCH(GGML_TYPE_Q6_K);    break;
        case GGML_TYPE_Q8_0:    MMF16_LAUNCH(GGML_TYPE_Q8_0);    break;
#undef MMF16_LAUNCH
#undef MMF16_LAUNCH_P
#undef MMF16_LAUNCH_PD
        default: GGML_ABORT("unsupported type");
    }
}

// ---------------------------------------------------------------------------------------------------
// The expert path.
//
// MUL_MAT_ID reaches ggml_cuda_mul_mat_id, not ggml_cuda_mul_mat, so the GEMM above never saw a
// mixture-of-experts model's experts: they stayed on MMQ and are 60 percent of that model's prefill
// (logs/moe-decode-2026-09-20). ggml_cuda_launch_mm_ids_helper already builds what is needed, the same
// call MMQ and MMF make: a compact ordering of the used (token, slot) pairs sorted by expert, the
// activation column each compact slot reads, the destination column it writes, and the range of each
// expert. This launches the same kernel over that ordering, one expert per block in z.
//
// The column tile wants to be narrower here than on the dense path. At a 512-token batch with eight of
// 256 experts used, an expert sees about sixteen columns, so a 64-wide tile would be mostly padding;
// GGML_RDNA1_PKF16_ID_BN picks among 16, 32 and 64.
static int id_bn() {
    static const int v = [] {
        const char * e = getenv("GGML_RDNA1_PKF16_ID_BN");
        const int n = e ? atoi(e) : 32;
        return (n == 16 || n == 32 || n == 64) ? n : 32;
    }();
    return v;
}

bool ggml_cuda_mmf16_rdna1_id_supported(const ggml_tensor * src0, const ggml_tensor * src1,
                                        const ggml_tensor * ids, const ggml_tensor * dst, const int cc) {
    static const bool enabled = [] { const char * e = getenv("GGML_RDNA1_PKF16_ID"); return e ? atoi(e) != 0 : true; }();
    if (!enabled || !mmf16_type_supported(src0, cc)) return false;
    if (!ids || ids->type != GGML_TYPE_I32) return false;
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguously_allocated(src0)) return false;
    if (src0->ne[0] % 256 != 0) return false;            // one 32-value group per stage, inside a superblock
    if (src0->ne[1] < mmf16_min_rows()) return false;
    // one weight matrix per expert, one batch, and the compact ordering the helper builds assumes these
    if (src0->ne[3] != 1 || src1->ne[3] != 1 || dst->ne[3] != 1) return false;
    if (src1->ne[2] != dst->ne[2] || ids->ne[1] != dst->ne[2]) return false;
    if (ids->ne[0] != dst->ne[1]) return false;
    if (src1->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) return false;
    if (src1->nb[1] % sizeof(float) != 0 || dst->nb[1] % sizeof(float) != 0) return false;
    if (src1->nb[2] % src1->nb[1] != 0) return false;
    // the token count is dimension 2 here, not dimension 1; below a few hundred compact columns MMQ wins
    // for the same reason it does on the dense path
    if (src1->ne[2] * ids->ne[0] < mmf16_min_cols()) return false;
    return true;
}

void ggml_cuda_mmf16_rdna1_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                              const ggml_tensor * ids, ggml_tensor * dst) {
    const int M = src0->ne[1], K = src0->ne[0];
    const int n_experts     = src0->ne[2];
    const int n_tokens      = ids->ne[1];
    const int n_expert_used = ids->ne[0];
    const int ne_get_rows   = n_tokens * n_expert_used;
    const size_t row_size   = ggml_row_size(src0->type, src0->ne[0]);
    cudaStream_t stream     = ctx.stream();

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_experts + 1);

    const int si1  = ids->nb[1] / ggml_element_size(ids);
    const int sis1 = src1->nb[2] / src1->nb[1];
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, n_tokens, n_expert_used, src1->ne[1], si1, sis1, /*write_inverse =*/ false, stream);
    CUDA_CHECK(cudaGetLastError());

    const char  * A = (const char *)  src0->data;
    const float * B = (const float *) src1->data;
    float       * C = (float *)       dst->data;
    const int sb  = src1->nb[1] / sizeof(float);     // stride between activation columns, in floats
    const int ldc = dst->nb[1]  / sizeof(float);     // stride between destination columns, in floats

    // four column tiles in flight per expert absorbs most routing skew; the kernel loops past that
    const dim3 grid((M + MMF16_BM - 1) / MMF16_BM, 4, n_experts);

#define MMF16_ID_LAUNCH(t, bn, tn)                                                                       \
    mmf16_rdna1_gemm<t, 1, false, true, MMF16_BM, bn, MMF16_TM, tn><<<grid, MMF16_BM, 0, stream>>>(      \
        A, B, C, M, ne_get_rows, K, row_size, sb, ldc,                                                   \
        ids_src1.get(), ids_dst.get(), expert_bounds.get(), (size_t) src0->nb[2])
#define MMF16_ID_BN(t)                                                                                   \
    do {                                                                                                 \
        const int bn = id_bn();                                                                          \
        if      (bn == 16) MMF16_ID_LAUNCH(t, 16, 2);                                                    \
        else if (bn == 64) MMF16_ID_LAUNCH(t, 64, 8);                                                    \
        else               MMF16_ID_LAUNCH(t, 32, 4);                                                    \
    } while (0)
    switch (src0->type) {
        case GGML_TYPE_Q4_K:    MMF16_ID_BN(GGML_TYPE_Q4_K);    break;
        case GGML_TYPE_Q5_K:    MMF16_ID_BN(GGML_TYPE_Q5_K);    break;
        case GGML_TYPE_Q6_K:    MMF16_ID_BN(GGML_TYPE_Q6_K);    break;
        case GGML_TYPE_Q8_0:    MMF16_ID_BN(GGML_TYPE_Q8_0);    break;
        case GGML_TYPE_IQ2_XXS: MMF16_ID_BN(GGML_TYPE_IQ2_XXS); break;
        case GGML_TYPE_IQ3_XXS: MMF16_ID_BN(GGML_TYPE_IQ3_XXS); break;
        case GGML_TYPE_IQ3_S:   MMF16_ID_BN(GGML_TYPE_IQ3_S);   break;
        case GGML_TYPE_IQ4_XS:  MMF16_ID_BN(GGML_TYPE_IQ4_XS);  break;
        default: GGML_ABORT("unsupported type");
    }
#undef MMF16_ID_BN
#undef MMF16_ID_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}
