#!/usr/bin/env bash
# Was the 14B comparison confounded by GGML_CUDA_DISABLE_GRAPHS=1?
# The 14B needs it at this depth (HIP graph instantiation fails at depth 12000),
# the 1.5B and 8B runs did not use it. So the three-model comparison varied two
# things at once. This runs the 8B with the flag, changing only that.
set -u
D=~/inv62; mkdir -p "$D"
HIP=~/llama-master/build-hip/bin
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
M8=/opt/models/qwen3-8b-q8_0.gguf
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }
log "=== 8B at d16128 WITH GGML_CUDA_DISABLE_GRAPHS=1 (the 14B condition)"
for i in $(seq 1 10); do
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_DISABLE_GRAPHS=1 LD_LIBRARY_PATH=$L \
    timeout -k 20 1800 $HIP/llama-bench -m $M8 -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 > "$D/g_$i.log" 2>&1
  log "  run $i: $(grep -aoE "tg8 @ d[0-9]+ \| +[0-9]+\.[0-9]+" "$D/g_$i.log" | grep -oE "[0-9.]+$" || echo FAIL)"
done
touch "$D/DONE"; log done
