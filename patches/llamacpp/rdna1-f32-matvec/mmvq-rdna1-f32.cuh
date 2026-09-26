#pragma once
#include "common.cuh"

bool ggml_cuda_mmvq_rdna1_f32_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
                                        const ggml_tensor * dst, int cc);
void ggml_cuda_mmvq_rdna1_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                              const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_device * fusion);
