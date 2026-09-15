#!/usr/bin/env bash
# Does the zeroed fp16 GEMM lose its scalars? Runs on the board.
# Instrumentation: scripts/apply_fp16_scalar_probe.py on ~/llama-master (see its header).
#
# Configuration is the defect's: qwen3-8B Q8_0, ctx 2048, two wikitext chunks, flash attention on,
# native gfx1013 rocBLAS, HSA_ENABLE_SDMA=0, as in scripts/defects_still_open.sh.
#
#   f32        reference, expected 9.0975
#   f16        the defect: a different wrong value each run, 11 to 24 on record
#   f16 + PTRMODE_DEVICE   scalars passed in device memory instead of by host pointer
#   f16 + TEMP_SENTINEL + TEMP             output stamped, beta 0: zeroed calls sum to 0
#   f16 + TEMP_SENTINEL + TEMP + BETA1     output stamped, beta 1: what a zeroed call leaves
#
# This harness counts no GPU faults; it reads perplexity and the instrumentation lines only.
set -u
D=~/s0915/fp16scalar; mkdir -p "$D"
H=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
export LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib HSA_ENABLE_SDMA=0
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }
run () { # run <label> <env...>
  local label=$1; shift
  env "$@" timeout -k 30 900 $H/llama-perplexity -m $M8 --no-mmap -ngl 99 -fa on -c 2048 \
    -f ~/wiki.test.raw --chunks 2 > "$D/$label.log" 2>&1
  log "  $label: rc=$? $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/$label.log" || echo FAIL) $(grep -a -m1 BC250PTRMODE "$D/$label.log")"
}
log "=== fp16 scalar probe, $(uname -r)"
run f32_ref       GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
run f16_base1     GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16
run f16_ptrdev1   GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 BC250_PTRMODE_DEVICE=1
run f16_base2     GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16
run f16_ptrdev2   GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 BC250_PTRMODE_DEVICE=1
run f16_ptrdev3   GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 BC250_PTRMODE_DEVICE=1
run f16_stamp_beta0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 BC250_TEMP_SENTINEL=1 BC250_TEMP=1
run f16_stamp_beta1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 BC250_TEMP_SENTINEL=1 BC250_TEMP=1 BC250_BETA1=1
for l in f16_stamp_beta0 f16_stamp_beta1; do
  grep -a "BC250TEMP " "$D/$l.log" | awk '{print NR, $3, $4}' > "$D/$l.sums.txt"
  log "  $l: $(wc -l < "$D/$l.sums.txt") GEMM outputs; zero sums at positions: $(awk '$3=="gemm_output_abs_sum=0" {printf "%s ", $1}' "$D/$l.sums.txt")"
done
log "=== done"; touch "$D/DONE"
