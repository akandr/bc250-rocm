#!/usr/bin/env bash
# Does the KQV defect need asynchronous graph execution?
#
# Two facts now point the same way. Identical rocBLAS call sequences return four
# different perplexities, so the variation is not in what is asked for. And
# running the same graph under llama-eval-callback, which synchronises and
# copies every tensor to the host after each op, gives six byte-identical dumps
# on a build whose perplexity runs vary across four values. Instrumentation that
# serialises execution makes the defect disappear.
#
# If that is what matters, disabling HIP graph capture should reduce or remove
# the variation too, since it changes how the work is submitted without touching
# the arithmetic. Graph capture was ruled out for the separate zeroed-fp16
# defect but has never been varied against this one.
#
# Six runs per arm, alternated rather than blocked.
set -u
exec 9>~/.inv102.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv102; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
SRC=~/llama-master
G=$SRC/src/llama-graph.cpp
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

cp "$G" "$D/llama-graph.cpp.orig"
log "=== removing the KQV precision line"
sed -i "/BC-250 test: f16 V aggregation in half precision corrupts at long context/,+1d" "$G"
cmake --build $SRC/build-hip -j6 > "$D/build.log" 2>&1
log "  build rc=$? set_prec(kqv lines now: $(grep -c 'set_prec(kqv' "$G")"

run () { # run <arm> <tag>
  local arm=$1 tag=$2
  if [ "$arm" = A ]; then
    env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L GGML_CUDA_NO_VMM=1 \
      timeout -k 30 1800 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
      -fa off -c 2048 -f $WIKI --chunks 2 > "$D/${tag}_$arm.log" 2>&1
  else
    env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L GGML_CUDA_NO_VMM=1 GGML_CUDA_DISABLE_GRAPHS=1 \
      timeout -k 30 1800 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
      -fa off -c 2048 -f $WIKI --chunks 2 > "$D/${tag}_$arm.log" 2>&1
  fi
  log "  $tag arm=$arm: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/${tag}_$arm.log" | grep -oE '[0-9.]+$' || echo FAIL)"
}

log "=== A default against B with graph capture disabled, six rounds, alternated"
for r in $(seq 1 6); do
  if [ $((r % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
  for a in $order; do run $a "r$r"; done
done

log "=== restoring the patch"
cp "$D/llama-graph.cpp.orig" "$G"
cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1
log "  rebuild rc=$? set_prec lines back: $(grep -c 'set_prec(kqv' "$G")"

log "=== summary (correct is 10.0103)"
for a in A B; do
  v=$(grep -oE "arm=$a: [0-9.]+" "$D/log" | awk '{print $2}')
  n=$(echo "$v" | grep -c .); ok=$(echo "$v" | grep -c "^10\.0103$"); dis=$(echo "$v" | sort -u | tr '\n' ' ')
  log "  arm $a: $ok of $n correct; distinct values: $dis"
done
touch "$D/DONE"; log done
