#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# probe_dispatch_boot.sh <tag>: does the bare probe's failure need more than one dispatch per
# invocation? The probe at 8.4M threads (32768 blocks) runs ten times with one dispatch per
# invocation, as the Fedora 43 factorial ran it, then three times with three dispatches, then five
# more times with one. Every element of every dispatch is checked (patches/compute_probe.c); the
# kernel log written during the battery is kept and searched as in hws_fullcheck_boot.sh.
set -u
TAG=${1:?tag}; O=$HOME/probedisp/$TAG; mkdir -p "$O"
B=$HOME/fullcheck
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/battery.log"; sync; }
{
  echo "date: $(date -Is)"; echo "kernel: $(uname -r)"; echo "cmdline: $(cat /proc/cmdline)"
  for p in bc250_flush_pasid_kiq bc250_flush_by_runlist bc250_cc_write_mode sched_policy gpu_recovery; do
    printf 'amdgpu.%s=%s\n' "$p" "$(cat /sys/module/amdgpu/parameters/$p 2>/dev/null)"; done
  echo "kfd $(grep -h simd_count /sys/class/kfd/kfd/topology/nodes/*/properties | sort -u | tail -1) (2 SIMDs per CU)"
  echo "HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-unset}"
} > "$O/header.txt"
sudo -n journalctl -k -b 0 --no-pager > "$O/kernel_log_before.txt"
run() { # name iters
  timeout -k 10 300 $B/compute_probe 32768 6000 $2 > "$O/$1.log" 2>&1; rc=$?
  log "  $1 dispatches=$2 rc=$rc $(grep -a -o 'iter [0-9]*:.*' "$O/$1.log" | awk '{print $2 $6}' | tr '\n' ' ') $(grep -a 'RESULT' "$O/$1.log" || echo 'NO RESULT')"
}
log "=== one dispatch per invocation, x10"
for i in $(seq -w 1 10); do run single_a$i 1; done
log "=== three dispatches per invocation, x3"
for i in 1 2 3; do run triple_$i 3; done
log "=== one dispatch per invocation, x5"
for i in 1 2 3 4 5; do run single_b$i 1; done
sudo -n journalctl -k -b 0 --no-pager > "$O/kernel_log.txt"
tail -n +$(( $(wc -l < "$O/kernel_log_before.txt") + 1 )) "$O/kernel_log.txt" > "$O/kernel_log_battery.txt"
rm -f "$O/kernel_log.txt" "$O/kernel_log_before.txt"
grep -i -E 'amdgpu|kfd|gfxhub|mmhub|drm' "$O/kernel_log_battery.txt" | \
  grep -i -E 'page fault|preempt|evict|unrecoverable|hang|timeout|time out|reset|test failed|error|MEMORY_APERTURE' \
  > "$O/kernel_log_summary.txt"
log "single-dispatch invocations with wrong elements: $(grep -l MISMATCH "$O"/single_*.log 2>/dev/null | wc -l)/15; triple invocations with wrong elements: $(grep -l MISMATCH "$O"/triple_*.log 2>/dev/null | wc -l)/3; amdgpu fault/failure lines: $(wc -l < "$O/kernel_log_summary.txt")"
touch "$O/DONE"
