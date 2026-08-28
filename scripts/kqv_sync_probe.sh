#!/usr/bin/env bash
# Re-run the synchronisation probe, this time proving it executed.
#
# The first attempt found that synchronising after every node did not remove the
# KQV defect. But the two arms ran in the same median time, 12 seconds each,
# which is not what serialising a graph of hundreds of nodes should look like,
# so the probe may never have run. A negative result from an instrument that did
# nothing is worth nothing, and this session has already been caught by that
# twice.
#
# Same patch, plus a counter printed at exit, so the log says how many
# synchronisations actually happened in each arm.
set -u
exec 9>~/.inv104.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv104; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
SRC=~/llama-master
G=$SRC/src/llama-graph.cpp
C=$SRC/ggml/src/ggml-cuda/ggml-cuda.cu
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

cp "$G" "$D/llama-graph.cpp.orig"; cp "$C" "$D/ggml-cuda.cu.orig"
log "=== removing the KQV line and adding a counted per-node sync"
sed -i "/BC-250 test: f16 V aggregation in half precision corrupts at long context/,+1d" "$G"
python3 - "$C" <<'PYEOF'
import sys
p=sys.argv[1]; s=open(p).read()
anchor = """                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);"""
add = """                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                {   // BC-250 probe: serialise without copying anything back
                    static const bool bc250_sync = getenv("BC250_SYNC_EACH_NODE") != nullptr;
                    static unsigned long bc250_n = 0;
                    if (bc250_sync) {
                        CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
                        if (++bc250_n % 20000 == 1) {
                            fprintf(stderr, "BC250SYNC count=%lu\\n", bc250_n);
                        }
                    }
                }"""
assert s.count(anchor)==1
open(p,"w").write(s.replace(anchor, add, 1))
PYEOF
cmake --build $SRC/build-hip -j6 > "$D/build.log" 2>&1
rc=$?; log "  build rc=$rc errors=$(grep -c 'error:' "$D/build.log")"
if [ "$rc" -ne 0 ]; then
  grep -m3 "error:" "$D/build.log" | tee -a "$D/log"
  cp "$D/llama-graph.cpp.orig" "$G"; cp "$D/ggml-cuda.cu.orig" "$C"
  cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1; touch "$D/DONE"; log done; exit 1
fi

run () { # run <arm> <tag>
  local arm=$1 tag=$2 extra=() t0 t1
  [ "$arm" = B ] && extra=(BC250_SYNC_EACH_NODE=1)
  t0=$(date +%s)
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L GGML_CUDA_NO_VMM=1 "${extra[@]}" \
    timeout -k 30 1800 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
    -fa off -c 2048 -f $WIKI --chunks 2 > "$D/${tag}_$arm.log" 2>&1
  t1=$(date +%s)
  log "  $tag arm=$arm: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/${tag}_$arm.log" | grep -oE '[0-9.]+$' || echo FAIL) wall=$((t1-t0))s syncs=$(grep -aoE 'BC250SYNC count=[0-9]+' "$D/${tag}_$arm.log" | tail -1 | grep -oE '[0-9]+$' || echo 0)"
}

log "=== A as normal against B synchronised after every node, six rounds, alternated"
for r in $(seq 1 6); do
  if [ $((r % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
  for a in $order; do run $a "r$r"; done
done

log "=== restoring both files"
cp "$D/llama-graph.cpp.orig" "$G"; cp "$D/ggml-cuda.cu.orig" "$C"
cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1
log "  rebuild rc=$? set_prec back: $(grep -c 'set_prec(kqv' "$G")  probe removed: $(grep -c BC250SYNC "$C")"

log "=== summary (correct is 10.0103)"
for a in A B; do
  v=$(grep -oE "arm=$a: [0-9.]+" "$D/log" | awk '{print $2}')
  w=$(grep -oE "arm=$a:.*wall=[0-9]+" "$D/log" | grep -oE 'wall=[0-9]+' | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END{print a[int(NR/2)+1]}')
  sy=$(grep -oE "arm=$a:.*syncs=[0-9]+" "$D/log" | grep -oE 'syncs=[0-9]+' | cut -d= -f2 | sort -n | tail -1)
  log "  arm $a: $(echo "$v" | grep -c '^10\.0103$') of $(echo "$v" | grep -c .) correct; median wall ${w}s; syncs observed ${sy}; distinct: $(echo "$v" | sort -u | tr '\n' ' ')"
done
touch "$D/DONE"; log done
