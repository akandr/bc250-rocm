#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Try to reproduce the fault of 22 September on demand.
#
# What triggered it, as far as the log says: repeated qwen2.5-1.5B decode at a deep primed context,
# one llama-bench invocation per point, alternating depth 16384 and 24576. It survived two full passes
# and faulted about one second into the third, and the first sweep of the same night had already
# produced two 28-percent dropouts at exactly those two depths. Nothing else in eight hours of soak at
# depth 0 provoked anything.
#
# Note the ambiguity this cannot resolve by itself: the fault is timestamped one second after the
# previous invocation logged its result, so it belongs either to the new process starting or to the
# previous one tearing down. Both are worth a look about and the loop exercises both.
#
# Stops at the first fault. Usage: fault_repro_depth.sh [max_passes]
set -u
N=${1:-40}
O=~/fault-repro; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
export HSA_ENABLE_SDMA=0
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }

base=$(faults)
log "start, faults already in this boot: $base, cmdline: $(cat /proc/cmdline)"
for p in $(seq 1 "$N"); do
  for d in 16384 24576; do
    r=$(LD_LIBRARY_PATH=$L timeout -k 20 600 "$HIP/llama-bench" -m /opt/models/qwen2.5-1.5b-q4km.gguf \
          -mmp 0 -ngl 99 -fa on -p 0 -n 64 -d $d -r 3 2>/dev/null |
        grep -aE "tg64" | awk -F'|' '{v=$(NF-1); gsub(/ /,"",v); print v}')
    n=$(faults)
    log "p$p d=$d tg64=${r:-FAIL} faults=$n"
    if [ "$n" -gt "$base" ]; then
      log "FAULTED after $((  (p-1)*2 + (d==16384 ? 1 : 2) )) invocations"
      sudo journalctl -b 0 -k --no-pager --since "-3 min" > "$O/journal.txt" 2>/dev/null
      touch "$O/DONE"; exit 0
    fi
  done
done
log "no fault in $((N*2)) invocations"
touch "$O/DONE"
