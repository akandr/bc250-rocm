#!/usr/bin/env bash
# Is kernel 7.1.5 actually required, or was the flush fix the whole story?
#
# The repo presents 7.1.5 as the working configuration and asks which change
# between 6.18 and 7.1.5 made compute survive. The churn sweep now runs clean
# on 6.18.9 once the map-side flush is present, which puts that framing in
# doubt. This runs the rest of the working-configuration checks on whichever
# kernel is booted, so the two can be compared directly.
set -u
D=~/inv47/$(uname -r); mkdir -p "$D"
export HSA_ENABLE_SDMA=0
NATIVE_ROCBLAS=$HOME/rocBLAS/build/release/rocblas-install/lib

log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/inv47.log"; sync; }

log "kernel=$(uname -r)"
log "simd=$(grep -h simd_count /sys/class/kfd/kfd/topology/nodes/*/properties | sort -u | tail -1)"
log "runlist=$(cat /sys/module/amdgpu/parameters/bc250_flush_by_runlist 2>/dev/null)"
log "kiq=$(cat /sys/module/amdgpu/parameters/bc250_flush_pasid_kiq 2>/dev/null)"
{
  echo "kernel=$(uname -r)"
  echo "simd=$(grep -h simd_count /sys/class/kfd/kfd/topology/nodes/*/properties | sort -u | tail -1)"
  echo "runlist=$(cat /sys/module/amdgpu/parameters/bc250_flush_by_runlist 2>/dev/null)"
} > "$D/config.txt"

# 1. bare compute probe, the historical liveness test
log "=== compute probe"
for nb in 4096 16384 32768; do
  timeout -k 10 300 ~/compute_probe $nb 6000 1 > "$D/probe_$nb.log" 2>&1
  rc=$?
  res=$(grep -aoE "ALL CORRECT|total_wrong=[0-9]+" "$D/probe_$nb.log" | tail -1)
  log "  nblocks=$nb rc=$rc ${res:-NO RESULT (hang)}"
  echo "probe_$nb rc=$rc ${res:-NORESULT}" >> "$D/result.txt"
done

# 2. native rocBLAS SGEMM, the historical wedge (cp queue preemption timeout)
log "=== SGEMM sweep, native gfx1013 rocBLAS"
for N in 256 512 1024 2048 4096; do
  LD_LIBRARY_PATH=$NATIVE_ROCBLAS timeout -k 15 600 ~/sgemm_iter $N 10 > "$D/sgemm_$N.log" 2>&1
  rc=$?
  fin=$(grep -aoE "DONE iters=[0-9]+ nwrong=[0-9]+" "$D/sgemm_$N.log" | tail -1)
  nw=$(grep -ac "WRONG" "$D/sgemm_$N.log")
  log "  N=$N rc=$rc ${fin:-NO DONE LINE (wedge)} wrong_lines=$nw"
  echo "sgemm_$N rc=$rc ${fin:-NODONE} wrong=$nw" >> "$D/result.txt"
done

# 3. llama.cpp perplexity gate against the established reference 8.9442
log "=== perplexity gate (reference 8.9442)"
timeout -k 30 2400 ~/llama-master/build-hip/bin/llama-perplexity \
  -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 \
  -f ~/wiki.test.raw --chunks 8 > "$D/ppl.log" 2>&1
rc=$?
ppl=$(grep -aoE "Final estimate: PPL = [0-9.]+" "$D/ppl.log" | tail -1)
log "  rc=$rc ${ppl:-NO PPL}"
echo "ppl rc=$rc ${ppl:-none}" >> "$D/result.txt"

# 4. sustained large dispatch, the case that historically wedged
log "=== sustained N=4096 x50"
LD_LIBRARY_PATH=$NATIVE_ROCBLAS timeout -k 15 900 ~/sgemm_iter 4096 50 > "$D/sustained.log" 2>&1
rc=$?
fin=$(grep -aoE "DONE iters=[0-9]+ nwrong=[0-9]+" "$D/sustained.log" | tail -1)
log "  rc=$rc ${fin:-NO DONE LINE (wedge)}"
echo "sustained rc=$rc ${fin:-NODONE}" >> "$D/result.txt"

log "=== faults in dmesg"
sudo dmesg | grep -ciE "memory access fault|preemption time out|ring.*timeout" > "$D/faults.txt"
log "  fault lines: $(cat "$D/faults.txt")"

touch "$D/DONE"
log "done"
