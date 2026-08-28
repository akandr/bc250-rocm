#!/usr/bin/env bash
# Did the callback suppress the defect, or did the workload not trigger it?
#
# Six eval-callback dumps of a 288-token prompt came back byte-identical, which
# has two readings: the callback synchronises and copies every tensor to the
# host and may have suppressed the defect, or a single 288-token batch simply
# does not trigger what 2048 tokens over two perplexity chunks does. Those must
# be separated before the null result means anything.
#
# Arm 1 is the control: the exact perplexity configuration known to show the
# defect, on this same unpatched build, six times. If it does not vary here,
# nothing else in this run is interpretable.
# Arm 2 repeats the dump comparison with a prompt long enough to span four
# ubatches instead of one, matching the perplexity workload much more closely.
set -u
exec 9>~/.inv100.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv100; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
SRC=~/llama-master
G=$SRC/src/llama-graph.cpp
P=/tmp/prompt_long.txt
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

cp "$G" "$D/llama-graph.cpp.orig"
log "=== removing the KQV precision line"
sed -i "/BC-250 test: f16 V aggregation in half precision corrupts at long context/,+1d" "$G"
cmake --build $SRC/build-hip -j6 > "$D/build.log" 2>&1
log "  build rc=$? set_prec(kqv lines now: $(grep -c 'set_prec(kqv' "$G")"

log "=== arm 1, control: the perplexity configuration that shows the defect, six runs"
for i in $(seq 1 6); do
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L GGML_CUDA_NO_VMM=1 \
    timeout -k 30 1800 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
    -fa off -c 2048 -f $WIKI --chunks 2 > "$D/ppl_$i.log" 2>&1
  log "  ppl run $i: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/ppl_$i.log" | grep -oE '[0-9.]+$' || echo FAIL)"
done
log "  distinct values: $(grep -oE 'ppl run [0-9]+: [0-9.]+' "$D/log" | awk '{print $4}' | sort -u | tr '\n' ' ')"

log "=== arm 2: dumps of a prompt spanning four ubatches"
head -c 9000 "$WIKI" > "$P"
for i in $(seq 1 6); do
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L GGML_CUDA_NO_VMM=1 \
    timeout -k 30 1800 $SRC/build-hip/bin/llama-eval-callback -m $Q15 -ngl 99 \
    -fa off -c 2048 -ub 512 -n 1 -f "$P" > "$D/dump_$i.txt" 2>&1
  log "  dump $i: rc=$? lines=$(wc -l < "$D/dump_$i.txt")"
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
      log "  dumps $a and $b: first differ at line $ln, inside tensor: ${t:-unknown}"
    fi
  done
done
log "  identical pairs: $same   differing pairs: $diffn"
touch "$D/DONE"; log done
