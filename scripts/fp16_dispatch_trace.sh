#!/usr/bin/env bash
# Does the zeroed fp16 GEMM pick a different rocBLAS kernel from the 35 that work?
#
# Established: exactly 36 fp16 cuBLAS calls happen per micro-batch, all value
# projections, and only the first of them ever returns zeros. The trigger is
# positional rather than a property of the shape, since all 36 share it. What has
# never been looked at is which Tensile solution rocBLAS selects for each call.
# If the failing one is dispatched to a different kernel than its 35 identical
# siblings, that is the localisation; if all 36 select the same kernel, the fault
# is in state around the call rather than in the kernel chosen.
#
# ROCBLAS_LAYER=6 combines the trace and the bench layer, which prints the
# solution actually used. Output is large, so this runs the smallest input that
# still reproduces the defect.
set -u
D=~/inv75; mkdir -p "$D"
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
WIKI=~/wiki.test.raw
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "=== run with fp16 compute and rocBLAS solution logging"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 LD_LIBRARY_PATH=$L \
    ROCBLAS_LAYER=6 \
    timeout -k 30 1800 $HIP/llama-perplexity -m $M8 --no-mmap -ngl 99 -fa on \
    -c 2048 -f $WIKI --chunks 2 > "$D/f16_trace.log" 2>&1
log "  ppl: $(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/f16_trace.log" | grep -oE "[0-9.]+$" || echo FAIL)"
log "  log size: $(wc -c < "$D/f16_trace.log") bytes"

log "=== the same with f32 compute, as the reference dispatch pattern"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    ROCBLAS_LAYER=6 \
    timeout -k 30 1800 $HIP/llama-perplexity -m $M8 --no-mmap -ngl 99 -fa on \
    -c 2048 -f $WIKI --chunks 2 > "$D/f32_trace.log" 2>&1
log "  ppl: $(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/f32_trace.log" | grep -oE "[0-9.]+$" || echo FAIL)"

log "=== what the fp16 run dispatched"
log "  distinct rocblas functions called: $(grep -aoE "^- \{ *.rocblas_function.: .[a-z_0-9]+" "$D/f16_trace.log" 2>/dev/null | sort -u | wc -l)"
grep -aoE "rocblas_function\": \"[a-z_0-9]+\"" "$D/f16_trace.log" 2>/dev/null | sort | uniq -c | sort -rn | head -6 | tee -a "$D/log"
log "  gemm_ex calls total: $(grep -ac "gemm_ex" "$D/f16_trace.log" 2>/dev/null)"
log "  distinct solution indices seen: $(grep -aoE "solution_index\": *[0-9]+" "$D/f16_trace.log" 2>/dev/null | sort -u | wc -l)"
grep -aoE "solution_index\": *[0-9]+" "$D/f16_trace.log" 2>/dev/null | sort | uniq -c | head -8 | tee -a "$D/log"
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
