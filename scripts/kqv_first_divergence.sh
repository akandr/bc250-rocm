#!/usr/bin/env bash
# Where does the KQV divergence begin? Second attempt at the dump comparison.
#
# The first attempt used a 288-token prompt and got six byte-identical dumps.
# The second used 9000 bytes of text and aborted on
# GGML_ASSERT(n_tokens_all <= cparams.n_batch), so all six dumps were identical
# crashes rather than identical results. This one sizes the prompt to span three
# ubatches of 512 while staying inside the batch limit.
#
# The control is already established: on this same unpatched build the
# perplexity configuration returns 1057.5494, 91.3845, 10.0103, 10.0103,
# 10.0103 and 115.8450 over six runs, so the defect is live and any null result
# here is about the instrument or the workload rather than the build.
set -u
exec 9>~/.inv101.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv101; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
SRC=~/llama-master
G=$SRC/src/llama-graph.cpp
P=/tmp/prompt_3ub.txt
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

head -c 6000 "$WIKI" > "$P"
cp "$G" "$D/llama-graph.cpp.orig"
log "=== removing the KQV precision line"
sed -i "/BC-250 test: f16 V aggregation in half precision corrupts at long context/,+1d" "$G"
cmake --build $SRC/build-hip -j6 > "$D/build.log" 2>&1
log "  build rc=$? set_prec(kqv lines now: $(grep -c 'set_prec(kqv' "$G")"

log "=== six dumps, prompt spanning three ubatches"
for i in $(seq 1 6); do
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L GGML_CUDA_NO_VMM=1 \
    timeout -k 30 1800 $SRC/build-hip/bin/llama-eval-callback -m $Q15 -ngl 99 \
    -fa off -c 2048 -b 2048 -ub 512 -n 1 -f "$P" > "$D/dump_$i.txt" 2>&1
  rc=$?
  log "  dump $i: rc=$rc lines=$(wc -l < "$D/dump_$i.txt") batches=$(grep -ac 'kqv-0 = ' "$D/dump_$i.txt")"
done

log "=== restoring the patch"
cp "$D/llama-graph.cpp.orig" "$G"
cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1
log "  rebuild rc=$? set_prec lines back: $(grep -c 'set_prec(kqv' "$G")"

log "=== first divergence between dump pairs"
same=0; diffn=0
for a in 1 2 3 4 5; do
  for b in $(seq $((a+1)) 6); do
    ln=$(cmp "$D/dump_$a.txt" "$D/dump_$b.txt" 2>/dev/null | grep -oE "line [0-9]+" | grep -oE "[0-9]+")
    if [ -z "$ln" ]; then same=$((same+1)); else
      diffn=$((diffn+1))
      t=$(head -n "$ln" "$D/dump_$a.txt" | grep -a "common_debug_cb_eval:" | tail -1 | sed -E 's/.*cb_eval: *//; s/ = .*//')
      log "  dumps $a and $b: first differ at line $ln, tensor: ${t:-unknown}"
    fi
  done
done
log "  identical pairs: $same   differing pairs: $diffn"
touch "$D/DONE"; log done
