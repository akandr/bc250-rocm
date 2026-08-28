#!/usr/bin/env bash
# Is it the submission granularity, or the host copy?
#
# Setting an eval callback makes the KQV defect vanish. Three of the things that
# path changes have been eliminated: graph capture, serialisation, and fusion.
# Two remain. The callback submits one node per graph launch instead of one
# launch per split, and it copies every tensor to the host.
#
# This isolates the first. The scheduler's own no-callback branch is patched to
# submit node by node, exactly as the callback branch does, but with no callback
# installed and nothing copied back. If the defect disappears, granularity is
# the mechanism and the host copy is irrelevant. If it survives, the host copy
# is what remains, which would be a stranger result and worth knowing.
set -u
exec 9>~/.inv107.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv107; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
SRC=~/llama-master
G=$SRC/src/llama-graph.cpp
B=$SRC/ggml/src/ggml-backend.cpp
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

cp "$G" "$D/llama-graph.cpp.orig"; cp "$B" "$D/ggml-backend.cpp.orig"
log "=== removing the KQV line and adding an opt-in per-node submission path"
sed -i "/BC-250 test: f16 V aggregation in half precision corrupts at long context/,+1d" "$G"
python3 - "$B" <<'PYEOF'
import sys
p=sys.argv[1]; s=open(p).read()
anchor = """        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {"""
add = """        if (!sched->callback_eval) {
            // BC-250 probe: submit one node per launch, as the callback branch
            // does, but with no callback and nothing copied to the host
            static const bool bc250_per_node = getenv("BC250_SPLIT_PER_NODE") != nullptr;
            static unsigned long bc250_launches = 0;
            if (bc250_per_node) {
                for (int j = 0; j < split->graph.n_nodes; j++) {
                    struct ggml_cgraph gv = ggml_graph_view(&split->graph, j, j + 1);
                    enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                    if (ec != GGML_STATUS_SUCCESS) {
                        return ec;
                    }
                    ggml_backend_synchronize(split_backend);
                    if (++bc250_launches % 50000 == 1) {
                        fprintf(stderr, "BC250PERNODE launches=%lu\\n", bc250_launches);
                    }
                }
            } else {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
            }
        } else {"""
assert s.count(anchor)==1, s.count(anchor)
open(p,"w").write(s.replace(anchor, add, 1))
PYEOF
cmake --build $SRC/build-hip -j6 > "$D/build.log" 2>&1
rc=$?; log "  build rc=$rc errors=$(grep -c 'error:' "$D/build.log")"
if [ "$rc" -ne 0 ]; then
  grep -m3 "error:" "$D/build.log" | tee -a "$D/log"
  cp "$D/llama-graph.cpp.orig" "$G"; cp "$D/ggml-backend.cpp.orig" "$B"
  cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1; touch "$D/DONE"; log done; exit 1
fi

run () { # run <arm> <tag>
  local arm=$1 tag=$2 extra=() t0 t1
  [ "$arm" = B ] && extra=(BC250_SPLIT_PER_NODE=1)
  t0=$(date +%s)
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L GGML_CUDA_NO_VMM=1 "${extra[@]}" \
    timeout -k 30 1800 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
    -fa off -c 2048 -f $WIKI --chunks 2 > "$D/${tag}_$arm.log" 2>&1
  t1=$(date +%s)
  log "  $tag arm=$arm: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/${tag}_$arm.log" | grep -oE '[0-9.]+$' || echo FAIL) wall=$((t1-t0))s launches=$(grep -aoE 'BC250PERNODE launches=[0-9]+' "$D/${tag}_$arm.log" | tail -1 | grep -oE '[0-9]+$' || echo 0)"
}

log "=== A as normal against B one node per launch, eight rounds, alternated"
for r in $(seq 1 8); do
  if [ $((r % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
  for a in $order; do run $a "r$r"; done
done

log "=== restoring both files"
cp "$D/llama-graph.cpp.orig" "$G"; cp "$D/ggml-backend.cpp.orig" "$B"
cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1
log "  rebuild rc=$? set_prec back: $(grep -c 'set_prec(kqv' "$G")  probe removed: $(grep -c BC250PERNODE "$B")"

log "=== summary (correct is 10.0103 on the normal path)"
for a in A B; do
  v=$(grep -oE "arm=$a: [0-9.]+" "$D/log" | awk '{print $2}')
  w=$(grep -oE "arm=$a:.*wall=[0-9]+" "$D/log" | grep -oE 'wall=[0-9]+' | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END{print a[int(NR/2)+1]}')
  lc=$(grep -oE "arm=$a:.*launches=[0-9]+" "$D/log" | grep -oE 'launches=[0-9]+' | cut -d= -f2 | sort -n | tail -1)
  log "  arm $a: $(echo "$v" | grep -c '^10\.0103$') of $(echo "$v" | grep -c .) at 10.0103; median wall ${w}s; launches seen ${lc}; distinct: $(echo "$v" | sort -u | tr '\n' ' ')"
done
touch "$D/DONE"; log done
