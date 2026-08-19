#!/usr/bin/env bash
# A SIGBUS in the ROCr AQL queue path, found while running a control.
#
# The control was meant to isolate GGML_CUDA_DISABLE_GRAPHS=1 as a confound in a
# variance comparison. Instead the second run died: SIGBUS in
# rocr::AMD::AqlQueue::StoreRelaxed, reached from hipStreamQuery through
# submitMarker and dispatchBarrierPacket, i.e. a store into the AQL queue that
# hit an invalid mapping. That is a host-side crash with no GPU fault logged.
#
# Three arms, all 8B at a primed depth of 16128, to separate flag from depth
# from chance. Each run is independent so a crash does not stop the arm.
set -u
D=~/inv63; mkdir -p "$D"
HIP=~/llama-master/build-hip/bin
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
M8=/opt/models/qwen3-8b-q8_0.gguf
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

run () { # run <tag> <depth> <extra env...>
  local tag=$1 depth=$2; shift 2
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L "$@" \
    timeout -k 20 1800 $HIP/llama-bench -m $M8 -ngl 99 -fa 1 -p 0 -n 8 -d "$depth" -r 1 \
    > "$D/${tag}.log" 2>&1
  local rc=$?
  local v; v=$(grep -aoE "tg8 @ d[0-9]+ \| +[0-9]+\.[0-9]+" "$D/${tag}.log" | grep -oE "[0-9.]+$")
  if [ -n "$v" ]; then log "  $tag: $v"; else log "  $tag: CRASH rc=$rc"; fi
}

log "=== A: graphs disabled, depth 16128 (the condition that crashed)"
for i in $(seq 1 10); do run "A_$i" 16128 GGML_CUDA_DISABLE_GRAPHS=1; done
log "=== B: graphs enabled, depth 16128 (same depth, flag removed)"
for i in $(seq 1 10); do run "B_$i" 16128; done
log "=== C: graphs disabled, depth 0 (same flag, no depth)"
for i in $(seq 1 6); do run "C_$i" 0 GGML_CUDA_DISABLE_GRAPHS=1; done
log "=== crash signatures this run"
coredumpctl list --no-pager 2>/dev/null | grep -c "llama-bench" | xargs -I{} log "  total llama-bench cores on the system: {}"
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
