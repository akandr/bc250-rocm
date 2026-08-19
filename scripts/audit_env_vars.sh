#!/usr/bin/env bash
# Check that every environment variable this work relies on is actually read by
# the library that would have to read it.
#
# This exists because two were not. GGML_CUDA_FORCE_MMQ and GGML_CUDA_FORCE_CUBLAS
# are compile-time options, so setting them in the environment does nothing, and
# both had been carried in command lines and cited in prose as though they
# selected a code path. One of them put a wrong label on a defect report for
# weeks. A variable that does nothing is invisible in exactly the way a variable
# that works is, so it needs checking rather than assuming.
#
# Run on the board. Paths are the ones this work uses; adjust for another setup.
set -u
HIPLIB=$(ls /usr/lib64/libamdhip64.so.6* 2>/dev/null | head -1)
HSALIB=$(ls /usr/lib64/libhsa-runtime64.so.1* 2>/dev/null | head -1)
BLASLIB=${ROCBLAS_LIB:-$HOME/rocBLAS/build/release/rocblas-install/lib/librocblas.so.4.4}
GGML=${GGML_LIB:-$HOME/llama-master/build-hip/bin/libggml-hip.so}

VARS="AMD_LOG_LEVEL GGML_CUDA_CUBLAS_COMPUTE_TYPE GGML_CUDA_DISABLE_GRAPHS
      GGML_CUDA_ENABLE_UNIFIED_MEMORY GGML_CUDA_FORCE_CUBLAS GGML_CUDA_FORCE_MMQ
      GPU_FORCE_BLIT_COPY_SIZE GPU_STAGING_BUFFER_SIZE HSA_ENABLE_INTERRUPT
      HSA_ENABLE_SDMA HSA_OVERRIDE_GFX_VERSION HSA_XNACK ROCBLAS_LAYER
      ROCBLAS_TENSILE_LIBPATH"

printf "%-34s %s\n" VARIABLE "read by"
for v in $VARS; do
  out=""
  for pair in "hip:$HIPLIB" "hsa:$HSALIB" "rocblas:$BLASLIB" "ggml:$GGML"; do
    n=${pair%%:*}; f=${pair#*:}
    [ -e "$f" ] && [ "$(strings "$f" 2>/dev/null | grep -cx "$v")" -gt 0 ] && out="$out $n"
  done
  printf "%-34s %s\n" "$v" "${out:-** NOT READ BY ANY OF THEM **}"
done
echo
echo "Expected: the two FORCE_ variables report not-read (they are compile-time)."
echo "Anything else reporting not-read is a variable being set for no effect."
