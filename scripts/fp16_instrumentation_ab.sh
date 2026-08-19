#!/usr/bin/env bash
# Does instrumentation suppress the zeroed fp16 GEMM, and what alpha/beta does
# llama.cpp actually pass?
#
# Two observations from the solution-logging run. First, with ROCBLAS_LAYER=6 the
# fp16 gate returned 9.5672, where the same configuration without logging returns
# 16 to 21. If that holds, the defect is perturbed by instrumentation, which is a
# real clue about its nature and a warning about every trace taken of it.
# Second, the rocBLAS bench trace prints alpha and beta as -0.00014782 for both,
# where the source passes half 1.0 and 0.0. That is probably a logging artifact
# rather than argument corruption, and the tree already carries a BC250_ALPHA
# hook that prints what llama.cpp holds, so it can be settled rather than guessed.
set -u
D=~/inv76; mkdir -p "$D"
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
WIKI=~/wiki.test.raw
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

gate () { # gate <tag> <extra env...>
  local tag=$1; shift
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 LD_LIBRARY_PATH=$L "$@" \
    timeout -k 30 1800 $HIP/llama-perplexity -m $M8 --no-mmap -ngl 99 -fa on \
    -c 2048 -f $WIKI --chunks 2 > "$D/$tag.log" 2>&1
  log "  $tag: $(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/$tag.log" | grep -oE "[0-9.]+$" || echo FAIL)"
}

log "=== does rocBLAS logging suppress the defect? alternated, four runs per arm"
for r in 1 2; do
  gate "nolog_r${r}a"
  gate "log_r${r}a"   ROCBLAS_LAYER=6
  gate "log_r${r}b"   ROCBLAS_LAYER=6
  gate "nolog_r${r}b"
done

log "=== what alpha and beta does llama.cpp actually pass on the fp16 path?"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 LD_LIBRARY_PATH=$L BC250_ALPHA=1 \
  timeout -k 30 900 $HIP/llama-perplexity -m $M8 --no-mmap -ngl 99 -fa on \
  -c 2048 -f $WIKI --chunks 1 > "$D/alpha.log" 2>&1
log "  distinct alpha/beta pairs seen: $(grep -aoE "alpha=[-0-9.e+]+ beta=[-0-9.e+]+" "$D/alpha.log" | sort -u | tr "\n" " ")"
log "  BC250ALPHA lines: $(grep -ac BC250ALPHA "$D/alpha.log")"
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
