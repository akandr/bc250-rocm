#pragma once
#include "common.cuh"

// RDNA1: prefill matmuls through a packed-fp16 tile GEMM instead of MMQ's emulated int8 tiles.
bool ggml_cuda_mmf16_rdna1_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void ggml_cuda_mmf16_rdna1(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// The same kernel over MUL_MAT_ID's experts, through the compact ordering mm_ids_helper builds.
bool ggml_cuda_mmf16_rdna1_id_supported(const ggml_tensor * src0, const ggml_tensor * src1,
                                        const ggml_tensor * ids, const ggml_tensor * dst, int cc);
void ggml_cuda_mmf16_rdna1_id(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                              const ggml_tensor * ids, ggml_tensor * dst);
