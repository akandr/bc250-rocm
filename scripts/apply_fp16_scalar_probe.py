#!/usr/bin/env python3
# Does the zeroed fp16 GEMM lose its scalars?
#
# What is established (INVESTIGATION.md, the fp16 section): the operands are intact on the device
# at the moment of the failing call, alpha and beta are correct bit patterns at the call site
# (0x3c00, 0x0000), the GEMM writes its output and writes zeros, only the first GEMM after a
# stretch of other GPU work fails, any GEMM placed in front of it absorbs the failure, the values
# vary run to run, and the system library's gfx1010 code objects do not show it.
#
# A kernel that received alpha = 0 and beta = 0 would produce exactly that output. Tensile
# kernels take the scalars as launch arguments, which on HIP travel separately from the operand
# buffers, and a first launch after other work is where a stale argument transfer would show.
# Two switches test it, both on the existing BC250 instrumentation in ggml-cuda.cu:
#
#   BC250_BETA1=1            pass beta = 1.0 instead of 0.0 on fp16-compute GEMMs; combine with
#                            BC250_TEMP_SENTINEL=1 (stamps the output) and BC250_TEMP=1 (sums it).
#                            Kernel sees both scalars: sum = |AB + stamp|. Sees beta, loses alpha:
#                            sum = |stamp|. Loses both: sum = 0.
#   BC250_PTRMODE_DEVICE=1   issue the same fp16 GEMM through rocblas_gemm_ex in device pointer
#                            mode, with alpha and beta held in device memory, so the scalars
#                            reach the kernel by a different route. The perplexity says whether
#                            the defect survives the change.
#
# BC250_BETA1 changes the numerics on purpose (it adds the stamp into the activations), so its
# perplexity is meaningless; read only the BC250TEMP sums. BC250_PTRMODE_DEVICE does not change
# the numerics, so its perplexity is the measurement.
#
# Usage on the board: apply_fp16_scalar_probe.py ~/llama-master/ggml/src/ggml-cuda/ggml-cuda.cu
import sys
p = sys.argv[1]
s = open(p).read()
if "BC250_BETA1" in s:
    print("already patched"); sys.exit(0)

old = "    const void * beta = traits::get_beta();\n"
assert s.count(old) == 1, "beta anchor"
new = old + '''    if (getenv("BC250_BETA1") && compute_type == GGML_TYPE_F16) {
        // built from the bit pattern: __float2half/__half2float are device functions and give
        // garbage when called on the host, which already misled one trace here
        static half bc250_one;
        const uint16_t one_bits = 0x3c00;
        memcpy(&bc250_one, &one_bits, sizeof(one_bits));
        beta = &bc250_one;
    }
'''
s = s.replace(old, new, 1)

old2 = '''        const char * bc250_sol = getenv("BC250_SOLUTION");
        if (bc250_sol && compute_type == GGML_TYPE_F16) {'''
assert old2 in s, "solution anchor"
new2 = '''        if (getenv("BC250_PTRMODE_DEVICE") && compute_type == GGML_TYPE_F16) {
            rocblas_handle rh = (rocblas_handle) ctx.cublas_handle();
            static half * dev_scalars = nullptr;
            if (!dev_scalars) {
                CUDA_CHECK(cudaMalloc(&dev_scalars, 2 * sizeof(half)));
            }
            const half host_scalars[2] = { *(const half *) alpha, *(const half *) beta };
            CUDA_CHECK(cudaMemcpy(dev_scalars, host_scalars, sizeof(host_scalars), cudaMemcpyHostToDevice));
            rocblas_set_pointer_mode(rh, rocblas_pointer_mode_device);
            rocblas_status st = rocblas_gemm_ex(rh,
                    rocblas_operation_transpose, rocblas_operation_none,
                    ne01, ne11, ne10,
                    dev_scalars, src0_ptr, rocblas_datatype_f16_r, s01,
                                 src1_ptr, rocblas_datatype_f16_r, s11,
                    dev_scalars + 1, dst_ptr, rocblas_datatype_f16_r, ne0,
                                     dst_ptr, rocblas_datatype_f16_r, ne0,
                    rocblas_datatype_f16_r, rocblas_gemm_algo_standard, 0, rocblas_gemm_flags_none);
            rocblas_set_pointer_mode(rh, rocblas_pointer_mode_host);
            static bool once = false;
            if (!once) {
                fprintf(stderr, "BC250PTRMODE device pointer mode status=%d\\n", (int) st);
                once = true;
            }
            GGML_ASSERT(st == rocblas_status_success);
        } else if (const char * bc250_sol = getenv("BC250_SOLUTION"); bc250_sol && compute_type == GGML_TYPE_F16) {'''
s = s.replace(old2, new2, 1)
open(p, "w").write(s)
print("patched", p)
