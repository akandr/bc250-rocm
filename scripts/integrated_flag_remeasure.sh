#!/usr/bin/env bash
# Re-measure the integrated-flag figure, the third cited from memory.
#
# The write-up and the upstream patch header both state that with upstream
# behaviour restored, perplexity reads 167 where 11.2 is correct. Extending the
# figure audit from the write-up to the patch headers found that 167 is backed
# by no surviving log, and it is the load-bearing number in a patch intended for
# upstream, so it is measured again rather than sent as remembered.
#
# Configuration taken from the log that backs the 11.2 half (inv2, ctx 512,
# batch 8, 8 chunks), not guessed. That log came from an older build (2da6686);
# this runs on the commit everything else here is measured at, 7ba604f, which
# also answers whether the patch is still needed there.
set -u
D=~/inv73; mkdir -p "$D"
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
SRC=~/llama-master
F=$SRC/ggml/src/ggml-cuda/ggml-cuda.cu
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "waiting for the queue to drain"
while [ ! -f ~/inv72/DONE ]; do sleep 60; done

ppl () { # ppl <tag>
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 30 3600 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
    -fa off -c 512 -b 8 -ub 8 -f $WIKI --chunks 8 > "$D/$1.log" 2>&1
  log "  $1: $(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/$1.log" | grep -oE "[0-9.]+$" || echo FAIL)"
}

cp "$F" "$D/ggml-cuda.cu.orig"
log "=== patched state (integrated forced false), the shipped configuration"
log "  current: $(grep -c "integrated = false; // BC-250" "$F") BC-250 line(s)"
ppl patched

log "=== reverting to upstream behaviour (integrated = prop.integrated)"
sed -i "s|info.devices\[id\].integrated = false; // BC-250 test: integrated path corrupts output on gfx1013|info.devices[id].integrated = prop.integrated;|" "$F"
log "  now: $(grep -c "integrated = prop.integrated" "$F") upstream line(s)"
cmake --build $SRC/build-hip -j6 > "$D/build.log" 2>&1
log "  build rc=$? errors=$(grep -c "error:" "$D/build.log")"
log "=== upstream behaviour (write-up and patch header say about 167)"
ppl upstream_1
ppl upstream_2

log "=== restoring the patch"
cp "$D/ggml-cuda.cu.orig" "$F"
cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1
log "  rebuild rc=$?"
ppl restored
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
