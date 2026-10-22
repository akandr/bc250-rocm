#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# sched2_boot.sh <tag>: the software-scheduler rows of the CU-by-scheduler table on the same stack
# and checks as hws_fullcheck_boot.sh, with short timeouts because this policy is expected to wedge.
# The probe (every element) at three sizes with three dispatches, the full-output rocBLAS check at
# N=256 and 1024, then the kernel log written during the battery is kept; under this policy the
# preemption-timeout message from kgd_hqd_destroy is countable, and the summary greps both
# schedulers' fault and failure messages as before.
set -u
TAG=${1:?tag}; O=$HOME/sched2/$TAG; mkdir -p "$O"
B=$HOME/fullcheck; L=/opt/bc250-rocm/lib64:/usr/lib64
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/battery.log"; sync; }
{
  echo "date: $(date -Is)"; echo "kernel: $(uname -r)"; echo "cmdline: $(cat /proc/cmdline)"
  for p in bc250_flush_pasid_kiq bc250_flush_by_runlist bc250_cc_write_mode sched_policy gpu_recovery; do
    printf 'amdgpu.%s=%s\n' "$p" "$(cat /sys/module/amdgpu/parameters/$p 2>/dev/null)"; done
  echo "kfd $(grep -h simd_count /sys/class/kfd/kfd/topology/nodes/*/properties | sort -u | tail -1) (2 SIMDs per CU)"
  echo "HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-unset}"
} > "$O/header.txt"
cat "$O/header.txt" >> "$O/battery.log"
sudo -n journalctl -k -b 0 --no-pager > "$O/kernel_log_before.txt"
log "=== bare probe, every element, three dispatches"
for nb in 4096 16384 32768; do
  timeout -k 10 120 $B/compute_probe $nb 6000 3 > "$O/probe_$nb.log" 2>&1; rc=$?
  log "  nblocks=$nb rc=$rc $(grep -a 'RESULT' "$O/probe_$nb.log" || echo 'NO RESULT (hang)')"
done
log "=== rocBLAS GEMM, every element"
for n in 256 1024; do
  LD_LIBRARY_PATH=$L timeout -k 10 120 $B/rocblas_fullcheck s nn $n 10 > "$O/gemm_s_nn_${n}_x10.log" 2>&1; rc=$?
  log "  s nn $n 10 rc=$rc $(grep -a '^DONE' "$O/gemm_s_nn_${n}_x10.log" || echo 'NO DONE LINE (wedge or fault)')"
done
sudo -n journalctl -k -b 0 --no-pager > "$O/kernel_log.txt"
tail -n +$(( $(wc -l < "$O/kernel_log_before.txt") + 1 )) "$O/kernel_log.txt" > "$O/kernel_log_battery.txt"
rm -f "$O/kernel_log.txt" "$O/kernel_log_before.txt"
grep -i -E 'amdgpu|kfd|gfxhub|mmhub|drm' "$O/kernel_log_battery.txt" | \
  grep -i -E 'page fault|preempt|evict|unrecoverable|hang|timeout|time out|reset|test failed|error|MEMORY_APERTURE' \
  > "$O/kernel_log_summary.txt"
log "preemption timeouts (cp queue preemption time out): $(grep -c 'cp queue preemption time out' "$O/kernel_log_battery.txt"); amdgpu fault/failure lines: $(wc -l < "$O/kernel_log_summary.txt")"
log "probes correct: $(grep -c 'ALL CORRECT' "$O"/probe_*.log | awk -F: '{s+=$2} END {print s}')/3, gemm runs all elements correct: $(grep -l ALL_ELEMENTS_CORRECT "$O"/gemm_*.log 2>/dev/null | wc -l)/2"
touch "$O/DONE"
