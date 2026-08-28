#!/usr/bin/env bash
# Verify the front-page claims that re-measurement has not yet reached.
#
# The throughput table, the correctness gates, the reproducer, PyTorch training
# and the open defects have all been re-checked on the current configuration.
# These have not: memory bandwidth, DGEMM, the context ceilings, the 27B prefill
# figure, and ROCm running concurrently with Vulkan. The board is kept powered
# off now, so they are batched into one powered window rather than run when each
# occurs to me.
#
# Not covered here, deliberately: the claim that a stock PyTorch wheel aborts.
# The only stock venv on this board had 56 gfx1013 Tensile files grafted into it
# during earlier work, and mistaking it for pristine has already produced one
# withdrawn result. Verifying that claim needs a fresh install, which is a
# download rather than a measurement.
set -u
exec 9>~/.frontpage.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv117; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
VK=~/llama-master/build-vk/bin
E=(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L)
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }

log "=== kernel $(uname -r), gpu_recovery=$(cat /sys/module/amdgpu/parameters/gpu_recovery)"

log "=== memory bandwidth (front page: 432 GB/s, 402 GiB/s, 0.12 percent across three runs)"
for i in 1 2 3; do
  r=$(LD_LIBRARY_PATH=$L timeout -k 20 300 ~/membw 2>&1 | grep -aoE "[0-9]+\.[0-9]+ *(GB/s|GiB/s)" | tr '\n' ' ')
  log "  run $i: ${r:-no output}"
done

log "=== DGEMM (front page: about 95 percent of the FP64 rate peak)"
r=$(LD_LIBRARY_PATH=$L timeout -k 20 600 ~/dgemm_iter 2>&1 | tail -3 | tr '\n' ' ')
log "  ${r:-no output}"

log "=== context ceilings, decode of 32 tokens with the cache primed"
for spec in "qwen2.5-1.5b-q4km.gguf|1.5B|8192" "qwen2.5-1.5b-q4km.gguf|1.5B|16384" \
            "qwen2.5-1.5b-q4km.gguf|1.5B|32768" "deepseek-r1-14b.gguf|ds14B|8192"; do
  f=${spec%%|*}; rest=${spec#*|}; name=${rest%%|*}; d=${rest#*|}
  "${E[@]}" timeout -k 30 3600 "$HIP/llama-bench" -m "/opt/models/$f" -lm mmap -ngl 99 -fa 1 \
    -p 0 -n 32 -d "$d" -r 1 > "$D/ctx_${name}_$d.log" 2>&1
  v=$(grep -aoE "tg32 @ d[0-9]+ \| +[0-9.]+" "$D/ctx_${name}_$d.log" | grep -oE "[0-9.]+$")
  log "  $name at depth $d: ${v:-FAIL}"
done

log "=== 27B prefill at a 16384-token prompt (front page: 41.4 t/s)"
"${E[@]}" timeout -k 60 5400 "$HIP/llama-bench" -m /opt/models/qwen3.8-27b-iq3xxs.gguf -lm mmap \
  -ngl 99 -fa 1 -p 16384 -n 0 -r 1 > "$D/prefill27b.log" 2>&1
log "  $(grep -aoE 'pp16384 \| +[0-9.]+' "$D/prefill27b.log" | tail -1 || echo FAIL)"

log "=== ROCm and Vulkan concurrently (front page: 806.1 ROCm, 1843.0 Vulkan)"
"${E[@]}" timeout -k 30 1800 "$HIP/llama-bench" -m /opt/models/qwen2.5-1.5b-q4km.gguf -mmp 0 \
  -ngl 99 -fa on -p 512 -n 0 -r 2 > "$D/conc_hip.log" 2>&1 &
h=$!
timeout -k 30 1800 "$VK/llama-bench" -m /opt/models/qwen2.5-1.5b-q4km.gguf -mmp 0 \
  -ngl 99 -fa on -p 512 -n 0 -r 2 > "$D/conc_vk.log" 2>&1 &
v=$!
wait $h $v 2>/dev/null
log "  ROCm   $(grep -aoE 'pp512 \| +[0-9.]+' "$D/conc_hip.log" | tail -1 || echo FAIL)"
log "  Vulkan $(grep -aoE 'pp512 \| +[0-9.]+' "$D/conc_vk.log" | tail -1 || echo FAIL)"

log "=== faults during this window: $(sudo journalctl -b 0 --no-pager 2>/dev/null | grep -cE 'page fault \(src_id|GPU reset begin')"
touch "$D/DONE"; log done
