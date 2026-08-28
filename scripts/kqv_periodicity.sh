#!/usr/bin/env bash
# Is the KQV corruption periodic across processes, or just intermittent?
#
# The context ladder ran three invocations at each of three contexts and the
# catastrophic run was the third one every time, positions 3, 6 and 9 of nine.
# If runs failed independently at the observed rate, landing on the last of each
# group three times running is about a 0.3 percent coincidence. But position and
# context were confounded there, since the ladder changed context every three
# runs, so the pattern may belong to either.
#
# Twelve consecutive identical invocations at one context separate them. If the
# failures land at 3, 6, 9 and 12 it is positional and something accumulates
# across processes. If they scatter, it is intermittent and the ladder pattern
# was chance.
set -u
exec 9>~/.inv96.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv96; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
SRC=~/llama-master
G=$SRC/src/llama-graph.cpp
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "waiting for the queue to drain"
while [ ! -f ~/inv95/DONE ]; do sleep 120; done

cp "$G" "$D/llama-graph.cpp.orig"
log "=== removing the KQV precision line"
sed -i "/BC-250 test: f16 V aggregation in half precision corrupts at long context/,+1d" "$G"
cmake --build $SRC/build-hip -j6 > "$D/build.log" 2>&1
log "  build rc=$? set_prec(kqv lines now: $(grep -c 'set_prec(kqv' "$G")"

log "=== twelve consecutive identical runs, ctx 2048"
for i in $(seq 1 12); do
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L GGML_CUDA_NO_VMM=1 \
    timeout -k 30 1800 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
    -fa off -c 2048 -f $WIKI --chunks 2 > "$D/run_$i.log" 2>&1
  log "  run $i: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/run_$i.log" | grep -oE '[0-9.]+$' || echo FAIL)"
done

log "=== restoring the patch"
cp "$D/llama-graph.cpp.orig" "$G"
cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1
log "  rebuild rc=$? set_prec lines back: $(grep -c 'set_prec(kqv' "$G")"

log "=== summary: which positions were wrong (correct is near 10.01)"
awk '/run [0-9]+:/{n=$3; sub(":","",n); v=$4; if (v+0 > 11 || v=="FAIL") printf "  position %s WRONG (%s)\n", n, v}' "$D/log" | tee -a "$D/log"
touch "$D/DONE"; log done
