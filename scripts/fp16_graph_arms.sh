#!/usr/bin/env bash
# The zeroed fp16 GEMM: angles not yet tried.
#
# Known: the first fp16 value-projection of a graph intermittently returns all
# zeros with both operands verifiably intact; it needs a running model (no
# standalone reproducer, and PyTorch on the same rocBLAS cannot reproduce it);
# models whose value weights are quantized never take the path.
#
# Untested until now, and the reason for this script: whether HIP graph capture
# is involved. Every observation so far comes from runs with graph capture on.
# If the defect disappears with GGML_CUDA_DISABLE_GRAPHS=1 it belongs to the
# capture-and-replay machinery rather than to rocBLAS or the hardware, which
# would be the single most useful thing left to learn about it.
set -u
D=~/fp16-graphs; mkdir -p "$D"
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
Q14=/opt/models/qwen3-14b.gguf
WIKI=~/wiki.test.raw
REPS=${1:-6}

log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

# The defect shows as a perplexity far above the reference rather than as a
# crash, so each arm is judged by its gate value. Reference: 8B 7.3503.
arm () { # arm <tag> <extra env assignments...>
  local tag=$1; shift
  local bad=0 vals=""
  for i in $(seq 1 "$REPS"); do
    env HSA_ENABLE_SDMA=0 \
        LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib \
        "$@" timeout -k 20 1800 "$HIP/llama-perplexity" -m "$M8" --no-mmap -ngl 99 -fa on \
        -c 2048 -f "$WIKI" --chunks 2 > "$D/${tag}_$i.log" 2>&1
    v=$(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/${tag}_$i.log" | grep -oE "[0-9.]+$")
    vals="$vals ${v:-FAIL}"
    case "$v" in 7.3*) ;; *) bad=$((bad+1)) ;; esac
  done
  log "  $tag: $vals   (off-reference: $bad of $REPS)"
}

log "=== A: fp16 compute type, graphs ON (the configuration the defect was seen in)"
arm fp16_graphson GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16

log "=== B: fp16 compute type, graphs OFF (the new angle)"
arm fp16_graphsoff GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 GGML_CUDA_DISABLE_GRAPHS=1

log "=== C: f32 compute type, graphs ON (the documented workaround, as a control)"
arm f32_graphson GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32

log "=== D: quantized-V model with fp16 compute, graphs ON (should never take the path)"
for i in $(seq 1 3); do
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 \
      LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib \
      timeout -k 20 1800 "$HIP/llama-perplexity" -m "$Q14" --no-mmap -ngl 99 -fa on \
      -c 2048 -f "$WIKI" --chunks 2 > "$D/q14_$i.log" 2>&1
  v=$(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/q14_$i.log" | grep -oE "[0-9.]+$")
  log "  q14 run $i: ${v:-FAIL}"
done

log "=== dmesg faults: $(sudo dmesg | grep -ciE 'memory access fault|preemption time out')"
touch "$D/DONE"; log done
