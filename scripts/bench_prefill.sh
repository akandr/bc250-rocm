#!/bin/bash
# ROCm (patched module, native gfx1013 + FA + MMQ) vs Vulkan, PREFILL benchmark.
# ROCm decode faults, so ROCm = prefill only; Vulkan = pp + tg for the full reference.
# Same board, same models, llama-bench, fresh boot.
set -u
L=~/sweeplogs/bench_prefill; mkdir -p $L
exec > >(tee $L/bench.log) 2>&1
HIP=$HOME/llama.cpp/build-hip/bin      # native gfx1013
VK=$HOME/llama.cpp/build-vulkan/bin
M=/opt/models/qwen2.5-1.5b-q4km.gguf
sudo systemctl stop ollama queue-runner 2>/dev/null; sleep 2
echo "=== bench_prefill $(date +%FT%T)"
grep -m1 flush_pasid /usr/src/linux-6.18.9/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c
journalctl -k -b 0 --no-pager | grep -m1 active_cu_number | grep -o active_cu.*

echo "######## ROCm prefill (native gfx1013 + FA + MMQ), pp128/pp256/pp512, r3"
for pp in 128 256 512; do
  HSA_ENABLE_SDMA=0 GGML_CUDA_FORCE_MMQ=1 LD_LIBRARY_PATH=$HIP \
    timeout -k 5 180 $HIP/llama-bench -m $M -ngl 999 -fa 1 -p $pp -n 0 -r 3 -mmp 0 2>&1 | grep -E "qwen|pp[0-9]"
  echo "  (pp$pp ROCm rc=${PIPESTATUS[0]})"
done

echo "######## Vulkan pp128/256/512 + tg128 (reference), r3"
for pp in 128 256 512; do
  timeout -k 5 180 $VK/llama-bench -m $M -ngl 999 -fa 1 -p $pp -n 0 -r 3 -mmp 0 2>&1 | grep -E "qwen|pp[0-9]"
done
timeout -k 5 180 $VK/llama-bench -m $M -ngl 999 -fa 1 -p 0 -n 128 -r 3 -mmp 0 2>&1 | grep -E "qwen|tg[0-9]"

echo "=== bench_prefill done $(date +%FT%T)"
