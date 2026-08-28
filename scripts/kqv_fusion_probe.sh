#!/usr/bin/env bash
# Is operator fusion involved in the KQV defect?
#
# Reading what llama-eval-callback actually changes, in ggml-backend.cpp, makes
# this the sharpest remaining candidate. With an eval callback set, the
# scheduler stops submitting the whole split as one graph and instead submits
# one node at a time through ggml_graph_view, synchronising between each. Two
# things therefore differ from the normal path, not one: the synchronisation,
# and the fact that no node ever sees its successors, so cross-node fusion
# cannot happen.
#
# Synchronisation has already been tested on its own and does nothing. Fusion
# has not. This disables it directly, behind an environment variable so both
# arms come from one binary, by making ggml_cuda_can_fuse return false. A
# counter proves the switch was live, since a probe that silently does nothing
# has already caught this session out once.
set -u
exec 9>~/.inv105.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv105; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
SRC=~/llama-master
G=$SRC/src/llama-graph.cpp
C=$SRC/ggml/src/ggml-cuda/ggml-cuda.cu
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

cp "$G" "$D/llama-graph.cpp.orig"; cp "$C" "$D/ggml-cuda.cu.orig"
log "=== removing the KQV line and adding an opt-in fusion kill switch"
sed -i "/BC-250 test: f16 V aggregation in half precision corrupts at long context/,+1d" "$G"
python3 - "$C" <<'PYEOF'
import sys
p=sys.argv[1]; s=open(p).read()
anchor = """                               std::initializer_list<enum ggml_unary_op> unary_ops) {"""
add = """                               std::initializer_list<enum ggml_unary_op> unary_ops) {
    {   // BC-250 probe: refuse every fusion when asked, and say how often
        static const bool bc250_nofuse = getenv("BC250_NO_FUSION") != nullptr;
        static unsigned long bc250_calls = 0;
        if (bc250_nofuse) {
            if (++bc250_calls % 50000 == 1) {
                fprintf(stderr, "BC250NOFUSE refusals=%lu\\n", bc250_calls);
            }
            return false;
        }
    }"""
assert s.count(anchor)==1, s.count(anchor)
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
  [ "$arm" = B ] && extra=(BC250_NO_FUSION=1)
  t0=$(date +%s)
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L GGML_CUDA_NO_VMM=1 "${extra[@]}" \
    timeout -k 30 1800 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
    -fa off -c 2048 -f $WIKI --chunks 2 > "$D/${tag}_$arm.log" 2>&1
  t1=$(date +%s)
  log "  $tag arm=$arm: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/${tag}_$arm.log" | grep -oE '[0-9.]+$' || echo FAIL) wall=$((t1-t0))s refusals=$(grep -aoE 'BC250NOFUSE refusals=[0-9]+' "$D/${tag}_$arm.log" | tail -1 | grep -oE '[0-9]+$' || echo 0)"
}

log "=== A as normal against B with fusion refused, eight rounds, alternated"
for r in $(seq 1 8); do
  if [ $((r % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
  for a in $order; do run $a "r$r"; done
done

log "=== restoring both files"
cp "$D/llama-graph.cpp.orig" "$G"; cp "$D/ggml-cuda.cu.orig" "$C"
cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1
log "  rebuild rc=$? set_prec back: $(grep -c 'set_prec(kqv' "$G")  probe removed: $(grep -c BC250NOFUSE "$C")"

log "=== summary (correct is 10.0103)"
for a in A B; do
  v=$(grep -oE "arm=$a: [0-9.]+" "$D/log" | awk '{print $2}')
  w=$(grep -oE "arm=$a:.*wall=[0-9]+" "$D/log" | grep -oE 'wall=[0-9]+' | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END{print a[int(NR/2)+1]}')
  rf=$(grep -oE "arm=$a:.*refusals=[0-9]+" "$D/log" | grep -oE 'refusals=[0-9]+' | cut -d= -f2 | sort -n | tail -1)
  log "  arm $a: $(echo "$v" | grep -c '^10\.0103$') of $(echo "$v" | grep -c .) correct; median wall ${w}s; refusals seen ${rf}; distinct: $(echo "$v" | sort -u | tr '\n' ' ')"
done
touch "$D/DONE"; log done
