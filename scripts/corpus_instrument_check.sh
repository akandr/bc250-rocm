#!/usr/bin/env bash
# Two constants that have never been varied in any measurement here.
#
# 1. Every correctness gate in this work uses one file, wiki.test.raw. Ten of
#    ten. The claim that ROCm and Vulkan agree, which is the backbone of the
#    correctness argument, has therefore only ever been tested on one text.
# 2. Essentially every throughput figure comes from one instrument, llama-bench,
#    forty uses against three for llama-cli. A systematic error in how it primes
#    or times would be invisible.
#
# This varies both. Two additional corpora: a different slice of the same wiki
# text, and concatenated C++ source, which is a large distribution shift. Then
# the same decode measurement through a second instrument.
set -u
D=~/inv70; mkdir -p "$D"
HIP=~/llama-master/build-hip/bin
VK=~/llama-master/build-vk/bin
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
M8=/opt/models/qwen3-8b-q8_0.gguf
E=(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L)
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "waiting for the previous job"
while [ ! -f ~/inv69/DONE ]; do sleep 60; done

log "=== building two more corpora"
tail -2000 ~/wiki.test.raw > "$D/wiki_tail.raw"
find ~/llama-master/src ~/llama-master/common -name "*.cpp" -o -name "*.h" 2>/dev/null | head -40 | xargs cat > "$D/code.txt" 2>/dev/null
for f in ~/wiki.test.raw "$D/wiki_tail.raw" "$D/code.txt"; do
  log "  $(basename $f): $(wc -c < $f) bytes"
done

gate () { # gate <tag> <backend-bin> <model> <corpus> <extra env...>
  local tag=$1 bin=$2 m=$3 c=$4; shift 4
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L "$@" \
    timeout -k 30 3600 "$bin/llama-perplexity" -m "$m" --no-mmap -ngl 99 -fa on \
    -c 2048 -f "$c" --chunks 4 > "$D/$tag.log" 2>&1
  log "  $tag: $(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/$tag.log" | grep -oE "[0-9.]+$" || echo FAIL)"
}

log "=== ROCm against Vulkan on three corpora, small model"
for c in wiki:$HOME/wiki.test.raw tail:$D/wiki_tail.raw code:$D/code.txt; do
  t=${c%%:*}; f=${c#*:}
  gate "q15_${t}_rocm" "$HIP" "$Q15" "$f" GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
  gate "q15_${t}_vk"   "$VK"  "$Q15" "$f"
done
log "=== the same on the 8B, and with fp16 compute, to see if the defect is corpus dependent"
for c in wiki:$HOME/wiki.test.raw tail:$D/wiki_tail.raw code:$D/code.txt; do
  t=${c%%:*}; f=${c#*:}
  gate "q8b_${t}_rocm_f32" "$HIP" "$M8" "$f" GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
  gate "q8b_${t}_vk"       "$VK"  "$M8" "$f"
  gate "q8b_${t}_rocm_f16" "$HIP" "$M8" "$f" GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16
done

log "=== second instrument: llama-cli decode rate against llama-bench, same model and config"
for i in 1 2 3; do
  "${E[@]}" timeout -k 20 900 $HIP/llama-bench -m $Q15 -ngl 99 -fa 1 -p 0 -n 64 -r 1 > "$D/inst_bench_$i.log" 2>&1
  log "  llama-bench tg64 run $i: $(grep -aoE "tg64 \| +[0-9]+\.[0-9]+" "$D/inst_bench_$i.log" | grep -oE "[0-9.]+$")"
done
for i in 1 2 3; do
  "${E[@]}" timeout -k 20 900 $HIP/llama-cli -m $Q15 -ngl 99 -fa on -no-cnv -st \
    -p "Write a short paragraph about the sea." -n 64 --temp 0 > "$D/inst_cli_$i.log" 2>&1
  log "  llama-cli eval rate run $i: $(grep -aoE "eval time =.*\(.*, +[0-9.]+ tokens per second\)" "$D/inst_cli_$i.log" | grep -oE "[0-9.]+ tokens" | grep -oE "[0-9.]+" | tail -1)"
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
