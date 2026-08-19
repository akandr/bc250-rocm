#!/usr/bin/env bash
# Corrected re-measurement of the KQV-precision figures.
#
# The first attempt (inv67) used --chunks 8 where the cited figures were
# measured over --chunks 2, and perplexity over a different amount of text is a
# different quantity, so it compared nothing. That is the same mistake as the
# fp16 chunk arm earlier today and it is recorded rather than quietly retried.
#
# The original harness (inv25_knobs.sh) is on the board and pins the
# configuration exactly: qwen2.5-1.5B Q4_K_M, ctx 4096, chunks 2, ub 512,
# -fa off, with GGML_CUDA_NO_VMM=1 set by its helper. Matched here, plus one
# arm without NO_VMM to check that constant is not doing anything either.
set -u
D=~/inv72; mkdir -p "$D"
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
SRC=~/llama-master
G=$SRC/src/llama-graph.cpp
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "waiting for the queue to drain"
while [ ! -f ~/inv71/DONE ]; do sleep 60; done

ppl () { # ppl <tag> <fa> <extra env...>
  local tag=$1 fa=$2; shift 2
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L "$@" \
    timeout -k 30 3600 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
    -fa "$fa" -c 4096 -f $WIKI --chunks 2 > "$D/$tag.log" 2>&1
  log "  $tag: $(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/$tag.log" | grep -oE "[0-9.]+$" || echo FAIL)"
}

cp "$G" "$D/llama-graph.cpp.orig"
log "=== removing the KQV precision line"
sed -i "/BC-250 test: f16 V aggregation in half precision corrupts at long context/,+1d" "$G"
log "  set_prec(kqv lines now: $(grep -c "set_prec(kqv" "$G")"
cmake --build $SRC/build-hip -j6 > "$D/build.log" 2>&1
log "  build rc=$? errors=$(grep -c "error:" "$D/build.log")"

log "=== patch removed, the exact configuration the figures came from"
log "    (write-up quotes 276.29 fa-off no-variable, 11.0631 fa-off with it, 11.0521 fa-on)"
ppl nopatch_faoff_novar off GGML_CUDA_NO_VMM=1
ppl nopatch_faoff_f32   off GGML_CUDA_NO_VMM=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
ppl nopatch_faon_f32    on  GGML_CUDA_NO_VMM=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
log "=== the same three without GGML_CUDA_NO_VMM, to check that constant is inert"
ppl nopatch_faoff_novar_novmm off
ppl nopatch_faoff_f32_novmm   off GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32

log "=== restoring the patch"
cp "$D/llama-graph.cpp.orig" "$G"
cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1
log "  rebuild rc=$? set_prec lines back: $(grep -c "set_prec(kqv" "$G")"
ppl restored_faoff_novar off GGML_CUDA_NO_VMM=1
ppl restored_faon_f32    on  GGML_CUDA_NO_VMM=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
log "=== vulkan reference, same configuration (write-up quotes 11.0279)"
timeout -k 30 900 ~/llama-master/build-vk/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
  -fa off -c 4096 -f $WIKI --chunks 2 > "$D/vk.log" 2>&1
log "  vulkan: $(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/vk.log" | grep -oE "[0-9.]+$" || echo FAIL)"
touch "$D/DONE"; log done
