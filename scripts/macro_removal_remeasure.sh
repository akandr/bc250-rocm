#!/usr/bin/env bash
# Re-measure the macro-removed arm, whose figure has no surviving log.
#
# The write-up states that with the gfx1013 entry removed from the RDNA1 macro,
# perplexity reads 8.9425 and looks healthy while generation returns garbage,
# which is the evidence for needing two different gates. That number appears in
# no log that survives, on the board or in the repository, so it is re-measured
# here rather than left cited from memory.
#
# Builds a second tree with only that one line reverted, so the comparison is
# against the working build with everything else identical.
set -u
D=~/inv65; mkdir -p "$D"
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
SRC=~/llama-master
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "waiting for the campaign to finish"
while [ ! -f ~/inv64/DONE ]; do sleep 60; done

H=$SRC/ggml/src/ggml-cuda/vendors/hip.h
cp "$H" "$D/hip.h.orig"
log "=== current macro line: $(grep -n "gfx1013__) *$" "$H" | head -1)"
sed -i "s/#if defined(__gfx1010__) || defined(__gfx1012__) || defined(__gfx1013__)/#if defined(__gfx1010__) || defined(__gfx1012__)/" "$H"
log "=== after removal: $(grep -n "define RDNA1" -B1 "$H" | head -2 | tr "\n" " ")"

log "=== rebuilding the HIP backend without the gfx1013 RDNA1 entry"
cmake --build $SRC/build-hip -j6 > "$D/build.log" 2>&1
log "  build rc=$? ($(grep -c "error:" "$D/build.log") errors)"

log "=== perplexity with the macro removed (write-up says about 8.94, healthy-looking)"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
  timeout -k 30 3600 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 -fa on \
  -c 4096 -f $WIKI --chunks 8 > "$D/ppl_nomacro.log" 2>&1
log "  ppl: $(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/ppl_nomacro.log" | grep -oE "[0-9.]+$" || echo FAIL)"

log "=== generation with the macro removed (write-up says garbled)"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
  timeout -k 20 600 $SRC/build-hip/bin/llama-cli -m $Q15 -ngl 99 -fa on -no-cnv \
  -p "The capital of France is" -n 24 --temp 0 > "$D/gen_nomacro.log" 2>&1
log "  generated: $(grep -a "capital of France" "$D/gen_nomacro.log" | tail -1 | cut -c1-120)"

log "=== restoring the macro and rebuilding"
cp "$D/hip.h.orig" "$H"
cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1
log "  rebuild rc=$?"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
  timeout -k 30 3600 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 -fa on \
  -c 4096 -f $WIKI --chunks 8 > "$D/ppl_restored.log" 2>&1
log "  ppl restored: $(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/ppl_restored.log" | grep -oE "[0-9.]+$" || echo FAIL)"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
  timeout -k 20 600 $SRC/build-hip/bin/llama-cli -m $Q15 -ngl 99 -fa on -no-cnv \
  -p "The capital of France is" -n 24 --temp 0 > "$D/gen_restored.log" 2>&1
log "  generated restored: $(grep -a "capital of France" "$D/gen_restored.log" | tail -1 | cut -c1-120)"
touch "$D/DONE"; log done
