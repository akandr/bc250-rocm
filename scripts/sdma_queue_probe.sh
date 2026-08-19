#!/usr/bin/env bash
# Does the SDMA engine consume the packet it is given, or never see it?
#
# Established: a host-to-device copy above 16384 bytes is handed to the SDMA
# engine by ROCr, never completes, and produces no trap interrupt, while the
# engine's interrupt path demonstrably works for kernel-submitted work and the
# user queue is created. What is not known is whether the engine consumed the
# packet and failed to signal, or never advanced at all.
#
# The KFD exposes queue descriptors in debugfs. Sampling the SDMA descriptor
# for the hanging process separates those two cases: a descriptor that changes
# means the queue was rung and moved, one that never changes means the work
# never reached the engine.
set -u
D=~/sdma-queue; mkdir -p "$D"

log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "=== baseline: SDMA queues visible in mqds with no HIP process running"
sudo cat /sys/kernel/debug/kfd/mqds 2>/dev/null | grep -ciE "sdma" | \
  xargs -I{} log "  sdma queue entries at rest: {}"

log "=== starting a copy that will hang (16385 bytes, SDMA enabled)"
HSA_ENABLE_SDMA=1 timeout -k 10 90 ~/sdma_probe_bytes > "$D/probe.log" 2>&1 &
PROBE=$!
sleep 12

log "=== sampling the queue descriptors during the hang"
for i in 1 2 3 4 5; do
  sudo cat /sys/kernel/debug/kfd/mqds > "$D/mqds_$i.txt" 2>/dev/null
  sudo cat /sys/kernel/debug/kfd/hqds > "$D/hqds_$i.txt" 2>/dev/null
  sleep 4
done

log "  sdma entries seen during the hang: $(grep -ciE 'sdma' "$D/mqds_1.txt" 2>/dev/null)"
log "  queue kinds present: $(grep -oiE '(compute|sdma)[a-z ]*queue' "$D/mqds_1.txt" 2>/dev/null | sort -u | tr '\n' ' ')"

log "=== did any descriptor change across the samples?"
for i in 2 3 4 5; do
  if diff -q "$D/mqds_1.txt" "$D/mqds_$i.txt" > /dev/null 2>&1; then
    log "  sample $i: identical to sample 1"
  else
    log "  sample $i: CHANGED from sample 1 ($(diff "$D/mqds_1.txt" "$D/mqds_$i.txt" | grep -c '^[<>]') differing lines)"
  fi
done

wait $PROBE 2>/dev/null
log "=== probe outcome: $(grep -aoE 'HUNG:.*|ALL SIZES COMPLETED' "$D/probe.log" | tail -1)"
log "=== rls (runlists) during the hang, for context"
sudo cat /sys/kernel/debug/kfd/rls > "$D/rls.txt" 2>/dev/null
log "  runlist entries: $(grep -c . "$D/rls.txt" 2>/dev/null)"
touch "$D/DONE"; log done
