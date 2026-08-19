#!/usr/bin/env bash
# aba_mapflush_verify.sh - A/B/A test of the map-side runlist flush.
# Param is runtime-writable: 1=unmap-only (old), 3=unmap+map (candidate fix).
# Clean boot required. Order: fix ON first (clean boot), then OFF (expect
# faults), then ON again (expect clean) - A/B/A so a fault-degraded boot
# cannot masquerade as the fix failing.
set -u
D=~/inv28v; mkdir -p $D
B=~/llama-master/build-hip/bin/test-backend-ops
H=~/llama-master/build-hip/bin
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
DS14=/opt/models/deepseek-r1-14b.gguf
P=/sys/module/amdgpu/parameters/bc250_flush_by_runlist
export HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$HOME/rocBLAS/build/release/rocblas-install/lib
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a $D/verify.log; sync; }

tbo_round () { # tbo_round <tag> <n>
  local tag=$1 n=$2 faults=0
  for i in $(seq 1 $n); do
    timeout -k 10 900 $B perf -o MUL_MAT -b ROCm0 > $D/tbo_${tag}_$i.log 2>&1
    local rc=$?
    local f=$(grep -ac "Memory access fault" $D/tbo_${tag}_$i.log)
    [ "$f" -gt 0 ] && faults=$((faults+1))
    log "  tbo $tag run $i: rc=$rc fault=$f"
  done
  log "TBO $tag: $faults/$n faulted"
}

log "boot: $(uname -r), param=$(cat $P), CU=$(rocminfo 2>/dev/null | grep -m1 -A1 "Compute Unit" | tail -1 | tr -s " ")"
sudo dmesg | grep -m2 "bc250-40cu\|mapflush\|runlist" | tee -a $D/verify.log

log "=== A: fix ON (param=3), TBO churn x3"
echo 3 | sudo tee $P > /dev/null; log "param=$(cat $P)"
tbo_round on1 3

log "=== B: fix OFF (param=1), TBO churn x2 (expect faults)"
echo 1 | sudo tee $P > /dev/null; log "param=$(cat $P)"
tbo_round off 2

log "=== A2: fix ON again (param=3), TBO churn x3"
echo 3 | sudo tee $P > /dev/null; log "param=$(cat $P)"
tbo_round on2 3

log "=== C: correctness + perf with fix ON"
timeout -k 15 600 $H/llama-perplexity -m $Q15 --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 > $D/ppl_on.log 2>&1
log "ppl (expect 8.9442): $(grep -oE "Final estimate.*" $D/ppl_on.log)"
timeout -k 15 600 $H/llama-bench -m $Q15 -mmp 0 -ngl 99 -fa on -p 512,2048 -n 64 -r 3 > $D/bench_on.log 2>&1
log "bench: $(grep -oE "(pp[0-9]+|tg64) *\| *[0-9.]+ ± [0-9.]+" $D/bench_on.log | tr "\n" " ")"

log "=== D: 14B load time, param=3 vs param=1 (map-side rebuild cost)"
for pv in 3 1; do
  echo $pv | sudo tee $P > /dev/null
  for t in 1 2; do
    /usr/bin/time -o $D/load_${pv}_$t.time -f "%e" timeout -k 15 900 $H/llama-bench -m $DS14 -mmp 0 -ngl 99 -fa on -p 0 -n 8 -r 1 > $D/load_${pv}_$t.log 2>&1
    log "  14B load param=$pv trial=$t: rc=$? wall=$(cat $D/load_${pv}_$t.time 2>/dev/null)s"
  done
done
echo 3 | sudo tee $P > /dev/null

log "=== E: gen_reuse x10 with fix ON"
ok=0; for i in $(seq 1 10); do
  timeout -k 10 120 ~/gen_reuse > $D/gr$i.log 2>&1
  rc=$?; grep -aqE "FAULT|VIOLATION|fault" $D/gr$i.log || [ $rc -ne 0 ] || ok=$((ok+1))
done
log "gen_reuse: $ok/10 clean"

log "INV28 VERIFY DONE (param left at 3)"
echo done > $D/DONE
