#!/usr/bin/env bash
# Supply artifacts for the KQV context ladder, which an earlier write-up quoted and no log
# supports.
#
# An earlier write-up of the KQV precision patch carried a table showing the
# defect scaling with context: correct at 1024, about 174 at 2048, and 180 to
# 3764 across runs at 4096. The figure audit finds no log behind the 1024 value
# or the 4096 upper bound. The 2026-08-18 re-measurement covered ctx 4096 only,
# so the ladder itself has never been reproduced.
#
# Procedure copied from scripts/kqv_removal_remeasure.sh rather than reinvented,
# since that harness pins the configuration the original figures came from:
# qwen2.5-1.5B Q4_K_M, chunks 2, -fa off, GGML_CUDA_NO_VMM=1. Only the context
# varies here. Three runs per rung, because the 4096 figure is a range across
# runs and a single value cannot show that.
set -u
D=~/inv94; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
SRC=~/llama-master
G=$SRC/src/llama-graph.cpp
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "waiting for the queue to drain"
while [ ! -f ~/inv91/DONE ]; do sleep 120; done

ppl () { # ppl <tag> <ctx>
  local tag=$1 ctx=$2
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L GGML_CUDA_NO_VMM=1 \
    timeout -k 30 3600 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
    -fa off -c "$ctx" -f $WIKI --chunks 2 > "$D/$tag.log" 2>&1
  log "  $tag: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/$tag.log" | grep -oE '[0-9.]+$' || echo FAIL)"
}

cp "$G" "$D/llama-graph.cpp.orig"
log "=== removing the KQV precision line"
sed -i "/BC-250 test: f16 V aggregation in half precision corrupts at long context/,+1d" "$G"
log "  set_prec(kqv lines now: $(grep -c 'set_prec(kqv' "$G")"
cmake --build $SRC/build-hip -j6 > "$D/build.log" 2>&1
log "  build rc=$? errors=$(grep -c 'error:' "$D/build.log")"

log "=== the context ladder without the patch (earlier figures 9.8260 / ~174 / 180 to 3764)"
for c in 1024 2048 4096; do
  for i in 1 2 3; do ppl "nopatch_c${c}_$i" "$c"; done
done

log "=== restoring the patch"
cp "$D/llama-graph.cpp.orig" "$G"
cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1
log "  rebuild rc=$? set_prec lines back: $(grep -c 'set_prec(kqv' "$G")"
log "=== control: the same ladder with the patch in place, one run per rung"
for c in 1024 2048 4096; do ppl "restored_c${c}" "$c"; done

log "=== summary"
for c in 1024 2048 4096; do
  v=$(grep -oE "nopatch_c${c}_[0-9]: [0-9.]+" "$D/log" | cut -d" " -f2 | tr '\n' ' ')
  r=$(grep -oE "restored_c${c}: [0-9.]+" "$D/log" | cut -d" " -f2)
  log "  ctx $c: without patch [$v] with patch [$r]"
done
touch "$D/DONE"; log done
