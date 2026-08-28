#!/usr/bin/env bash
# Does the KQV corruption correspond to a different GEMM being dispatched?
#
# Twelve consecutive identical runs produced nine distinct perplexities, but the
# correct 10.0103 recurred three times and a wrong 115.8450 twice, both exact to
# four decimals. Accumulated rounding error does not reproduce a wrong answer
# exactly, so something is selecting between a small set of behaviours. Kernel
# selection is the obvious candidate and has never been checked.
#
# This logs every rocBLAS call through ROCBLAS_LAYER=2 for eight unpatched runs
# at one context, records the perplexity of each, and summarises each log by the
# distinct call signatures it contains. If a correct run and a wrong run differ
# in their signatures, selection is implicated. If they are identical, it is not,
# and the difference lies below rocBLAS's own interface.
#
# Tracing has already been shown not to suppress the related fp16 defect, so the
# instrument is unlikely to hide this one, but the perplexities recorded here are
# the check on that: if all eight come back correct under logging, the trace
# itself changed the outcome and the run says nothing.
set -u
exec 9>~/.inv98.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv98; mkdir -p "$D"
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

log "=== eight runs at ctx 2048 with every rocBLAS call logged"
for i in $(seq 1 8); do
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L GGML_CUDA_NO_VMM=1 \
      ROCBLAS_LAYER=2 ROCBLAS_LOG_BENCH_PATH="$D/bench_$i.txt" \
    timeout -k 30 1800 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
    -fa off -c 2048 -f $WIKI --chunks 2 > "$D/run_$i.log" 2>&1
  v=$(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/run_$i.log" | grep -oE '[0-9.]+$' || echo FAIL)
  # summarise the trace by distinct call signatures, dropping the pointer values
  if [ -f "$D/bench_$i.txt" ]; then
    sed -E 's/0x[0-9a-f]+/PTR/g' "$D/bench_$i.txt" | sort | uniq -c | sort -rn > "$D/sig_$i.txt"
    sigs=$(wc -l < "$D/sig_$i.txt"); calls=$(wc -l < "$D/bench_$i.txt")
  else
    sigs=0; calls=0
  fi
  log "  run $i: ppl=$v calls=$calls distinct_signatures=$sigs"
done

log "=== restoring the patch"
cp "$D/llama-graph.cpp.orig" "$G"
cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1
log "  rebuild rc=$? set_prec lines back: $(grep -c 'set_prec(kqv' "$G")"

log "=== comparing a correct run against a wrong one"
# Field indices are off by one from the obvious reading, because every log line
# carries a timestamp: "[08:23:31]   run 1: ppl=172.3724 calls=448 ...". The run
# number is $3 and the perplexity is $4. The first version used $2 and $3, so the
# run number came out as the literal word "run", no wrong run was ever found, and
# this whole comparison printed "cannot compare" in every invocation. The
# comparison the write-up reports was done by hand afterwards.
correct=$(awk '/run [0-9]+: ppl=10\.0103/{print $3; exit}' "$D/log" | tr -d ':')
wrong=$(awk '/run [0-9]+: ppl=/{v=$4; sub("ppl=","",v); if (v+0 > 11) {print $3; exit}}' "$D/log" | tr -d ':')
log "  correct run: ${correct:-none}   wrong run: ${wrong:-none}"
if [ -n "${correct:-}" ] && [ -n "${wrong:-}" ]; then
  if diff -q "$D/sig_$correct.txt" "$D/sig_$wrong.txt" > /dev/null 2>&1; then
    log "  SIGNATURES IDENTICAL: kernel selection at the rocBLAS interface is not the difference"
  else
    log "  SIGNATURES DIFFER: $(diff "$D/sig_$correct.txt" "$D/sig_$wrong.txt" | grep -c '^[<>]') differing lines"
    diff "$D/sig_$correct.txt" "$D/sig_$wrong.txt" | head -40 > "$D/signature_diff.txt"
  fi
else
  log "  cannot compare: need at least one correct and one wrong run"
fi
touch "$D/DONE"; log done
