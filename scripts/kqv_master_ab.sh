#!/usr/bin/env bash
# Does current llama.cpp master still need patches/llamacpp/0002 (KQV f32 precision)?
#
# Upstream #28576 moved the flash-attention MMA path to fp32 accumulation, which is
# near what 0002 addresses. This runs the configuration that exposed the defect
# (flash attention off, ctx 2048, two chunks, where the unpatched result varies
# run to run) four times with the KQV line and four times without it, in one boot,
# then restores the line.
#
# COMPUTE_TYPE sets GGML_CUDA_CUBLAS_COMPUTE_TYPE. Run it both ways: with f32 the
# GEMMs are forced to f32 and the defect cannot show, which is the first thing this
# script found when it was run with the variable set out of habit.
#   COMPUTE_TYPE=f32 D=~/s0914/kqv-master       bash kqv_master_ab.sh
#   COMPUTE_TYPE=    D=~/s0914/kqv-master-noenv bash kqv_master_ab.sh
set -u
SRC=${SRC:-~/llama-new}; D=${D:-~/s0914/kqv-master-noenv}; mkdir -p "$D"
G=$SRC/src/llama-graph.cpp
export LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib
[ -n "${COMPUTE_TYPE:-}" ] && export GGML_CUDA_CUBLAS_COMPUTE_TYPE=$COMPUTE_TYPE
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }
run () { for i in 1 2 3 4; do
  timeout -k 30 1800 "$SRC/build-hip/bin/llama-perplexity" -m /opt/models/qwen2.5-1.5b-q4km.gguf -lm none -ngl 99 \
    -fa off -c 2048 -f ~/wiki.test.raw --chunks 2 > "$D/$1_$i.log" 2>&1
  log "  $1 run $i: $(grep -aoE 'PPL = [0-9.]+' "$D/$1_$i.log" | tail -1 || echo FAIL)"; done; }
log "=== compute type '${COMPUTE_TYPE:-unset}'; with 0002 (set_prec lines: $(grep -c 'set_prec(kqv' "$G"))"; run with
cp "$G" "$D/llama-graph.cpp.with"
sed -i "/BC-250 test: f16 V aggregation in half precision corrupts at long context/,+1d" "$G"
cmake --build "$SRC/build-hip" -j12 --target llama-perplexity > "$D/build.log" 2>&1
log "=== without 0002 (build rc=$?, set_prec lines: $(grep -c 'set_prec(kqv' "$G"))"; run without
cp "$D/llama-graph.cpp.with" "$G"
cmake --build "$SRC/build-hip" -j12 --target llama-perplexity >> "$D/build.log" 2>&1
log "=== restored (rc=$?, set_prec lines: $(grep -c 'set_prec(kqv' "$G"))"
