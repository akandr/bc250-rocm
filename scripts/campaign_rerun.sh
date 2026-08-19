#!/usr/bin/env bash
# Re-run the five-model campaign on the CURRENT stack.
#
# The figures quoted throughout the write-up come from the campaign of
# 2026-08-12. Since then the patch scripts were corrected, the map-side flush
# was added, the kernel moved, and the library recipe settled. Those numbers are
# the most-cited data in the document and nothing has re-measured them, so this
# repeats the HIP arms exactly as the original harness ran them.
#
# Vulkan is not re-measured: it does not use any part of the ROCm stack that
# changed. One Vulkan decode point is taken as a spot check on the comparison.
set -u
D=~/inv64; mkdir -p "$D"
HIP=~/llama-master/build-hip/bin
VK=~/llama-master/build-vk/bin
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
WIKI=~/wiki.test.raw
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
Q8B=/opt/models/qwen3-8b-q8_0.gguf
DS14=/opt/models/deepseek-r1-14b.gguf
Q14=/opt/models/qwen3-14b.gguf
MOE=/opt/models/qwen3.6-35b-a3b-iq2m.gguf
E=(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L)
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }
pick () { grep -aoE "$2 \| +[0-9]+\.[0-9]+" "$1" | grep -oE "[0-9.]+$" | tr "\n" " "; }

log "waiting for the previous job"
while [ ! -f ~/inv63/DONE ]; do sleep 60; done
log "=== small model: prefill and decode (README quotes pp512 806, forced-BLAS 936, tg64 113.5)"
"${E[@]}" timeout -k 20 2400 $HIP/llama-bench -m $Q15 -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3 > "$D/q15.log" 2>&1
log "  q15 pp512: $(pick "$D/q15.log" pp512)  tg64: $(pick "$D/q15.log" tg64)"
"${E[@]}" GGML_CUDA_FORCE_CUBLAS=1 timeout -k 20 2400 $HIP/llama-bench -m $Q15 -mmp 0 -ngl 99 -fa on -p 512 -n 0 -r 3 > "$D/q15_blas.log" 2>&1
log "  q15 pp512 forced-BLAS: $(pick "$D/q15_blas.log" pp512)"
log "=== large models: pp512 and tg64 at depth 0"
for m in Q8B:$Q8B DS14:$DS14 Q14:$Q14 MOE:$MOE; do
  t=${m%%:*}; f=${m#*:}
  "${E[@]}" timeout -k 20 4800 $HIP/llama-bench -m $f -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 2 > "$D/$t.log" 2>&1
  log "  $t pp512: $(pick "$D/$t.log" pp512)  tg64: $(pick "$D/$t.log" tg64)"
done
log "=== perplexity gates (chunks 2, ctx 2048), the campaign gate condition"
for m in Q15:$Q15 Q8B:$Q8B DS14:$DS14 Q14:$Q14 MOE:$MOE; do
  t=${m%%:*}; f=${m#*:}
  "${E[@]}" timeout -k 30 3600 $HIP/llama-perplexity -m $f --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 2 > "$D/ppl_$t.log" 2>&1
  log "  $t ppl: $(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/ppl_$t.log" | grep -oE "[0-9.]+$" || echo FAIL)"
done
log "=== Vulkan spot check on the 8B decode (README quotes 39.1)"
timeout -k 20 3600 $VK/llama-bench -m $Q8B -mmp 0 -ngl 99 -fa on -p 0 -n 64 -r 2 > "$D/vk_8b.log" 2>&1
log "  vulkan 8B tg64: $(pick "$D/vk_8b.log" tg64)"
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
