#!/usr/bin/env bash
# Test the batch-boundary model of the zeroed fp16 GEMM by prediction.
#
# The zeros land on the first fp16 GEMM of every evaluated batch after the first:
# at context 2048 with the default ubatch of 512 there are four batches, 144 GEMM
# calls, and three zeros at calls 37, 73 and 109.
#
# If that model is right, the number of zeros is the number of batches minus one,
# and the spacing is the calls per batch. Predictions, written before running:
#
#   -ub 2048 -> one batch   -> 36 calls,  0 zeros
#   -ub 1024 -> two batches -> 72 calls,  1 zero  at call 37
#   -ub 256  -> eight batches -> 288 calls, 7 zeros at 19, 37, 55, 73, ...
#
# A single batch producing no zeros would also be a workaround worth knowing.
set -u
exec 9>~/.ubatch.lock; flock -n 9 || { echo locked; exit 1; }
D=~/inv125; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }

log "=== predicted: ub 2048 -> 0 zeros, ub 1024 -> 1, ub 512 -> 3, ub 256 -> 7"
for ub in 2048 1024 512 256; do
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 BC250_TEMP=1 LD_LIBRARY_PATH=$L \
    timeout -k 30 1800 "$HIP/llama-perplexity" -m $M8 --no-mmap -ngl 99 -fa on -c 2048 -ub $ub \
    -f ~/wiki.test.raw --chunks 1 > "$D/ub_$ub.log" 2>&1
  tot=$(grep -ac BC250TEMP "$D/ub_$ub.log")
  zer=$(grep -a BC250TEMP "$D/ub_$ub.log" | grep -c "abs_sum=0 ")
  pos=$(grep -a BC250TEMP "$D/ub_$ub.log" | grep -n "abs_sum=0 " | cut -d: -f1 | tr '\n' ' ')
  ppl=$(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/ub_$ub.log" | grep -oE '[0-9.]+$')
  log "  ub $ub: $tot calls, $zer zeros at [$pos] ppl=${ppl:-FAIL}"
done
touch "$D/DONE"; log done
