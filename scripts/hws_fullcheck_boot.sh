#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# hws_fullcheck_boot.sh <tag>: one boot's correctness battery under hardware scheduling, with checks
# that can fail. The bare probe (patches/compute_probe.c) zero-fills its buffer and compares every
# element; rocBLAS GEMM (patches/rocblas_fullcheck.cpp) fills its output with NaN before every call
# and compares every element exactly. The whole kernel log of the boot is saved raw (kernel_log.txt),
# the part written during the battery in kernel_log_battery.txt, and kernel_log_summary.txt greps
# that part's amdgpu/KFD lines for the fault and failure messages of both schedulers: GPU page faults,
# failed or timed-out queue preemption, failed eviction, the "unrecoverable state" warning, ring
# timeouts and resets, and amdgpu/KFD errors.
set -u
TAG=${1:?tag}; O=$HOME/hwsfull/$TAG; mkdir -p "$O"
B=$HOME/fullcheck; L=/opt/bc250-rocm/lib64:/usr/lib64
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/battery.log"; sync; }
{
  echo "date: $(date -Is)"; echo "kernel: $(uname -r)"; echo "cmdline: $(cat /proc/cmdline)"
  for p in bc250_flush_pasid_kiq bc250_flush_by_runlist bc250_cc_write_mode sched_policy gpu_recovery; do
    printf 'amdgpu.%s=%s\n' "$p" "$(cat /sys/module/amdgpu/parameters/$p 2>/dev/null)"; done
  echo "kfd $(grep -h simd_count /sys/class/kfd/kfd/topology/nodes/*/properties | sort -u | tail -1) (2 SIMDs per CU)"
  echo "rocblas: $(LD_LIBRARY_PATH=$L ldd $B/rocblas_fullcheck | grep rocblas)"
  echo "HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-unset}"
} > "$O/header.txt"
cat "$O/header.txt" >> "$O/battery.log"
sudo -n journalctl -k -b 0 --no-pager > "$O/kernel_log_before.txt"
log "=== bare probe, every element"
for nb in 4096 16384 32768; do
  timeout -k 10 300 $B/compute_probe $nb 6000 3 > "$O/probe_$nb.log" 2>&1; rc=$?
  log "  nblocks=$nb rc=$rc $(grep -a 'RESULT' "$O/probe_$nb.log" || echo 'NO RESULT (hang)')"
done
fc() { # type op N iters
  LD_LIBRARY_PATH=$L timeout -k 10 900 $B/rocblas_fullcheck "$@" > "$O/gemm_$1_$2_$3_x$4.log" 2>&1; rc=$?
  log "  $* rc=$rc $(grep -a '^DONE' "$O/gemm_$1_$2_$3_x$4.log" || echo 'NO DONE LINE (wedge or fault)') sclk=$(grep '\*' /sys/class/drm/card*/device/pp_dpm_sclk | cut -d: -f2 | tr -d ' *')"
}
log "=== rocBLAS GEMM, every element, output NaN-filled before each call"
for n in 256 512 1024 2048 4096; do fc s nn $n 10; done
fc s tn 4096 10; fc hs tn 4096 10; fc d nn 2048 10
log "=== sustained, every element of every iteration"
fc s nn 4096 200
fc hs tn 4096 200
log "=== bare probe after the battery"
timeout -k 10 300 $B/compute_probe 32768 6000 3 > "$O/probe_after.log" 2>&1
log "  rc=$? $(grep -a 'RESULT' "$O/probe_after.log" || echo 'NO RESULT (hang)')"
sudo -n journalctl -k -b 0 --no-pager > "$O/kernel_log.txt"
tail -n +$(( $(wc -l < "$O/kernel_log_before.txt") + 1 )) "$O/kernel_log.txt" > "$O/kernel_log_battery.txt"
grep -i -E 'amdgpu|kfd|gfxhub|mmhub|drm' "$O/kernel_log_battery.txt" | \
  grep -i -E 'page fault|preempt|evict|unrecoverable|hang|timeout|time out|reset|test failed|error|MEMORY_APERTURE' \
  > "$O/kernel_log_summary.txt"
log "kernel log lines during the battery: $(wc -l < "$O/kernel_log_battery.txt"), amdgpu fault/failure lines among them: $(wc -l < "$O/kernel_log_summary.txt")"
log "probes correct: $(grep -c 'ALL CORRECT' "$O"/probe_*.log | awk -F: '{s+=$2} END {print s}')/4, gemm runs all elements correct: $(grep -l ALL_ELEMENTS_CORRECT "$O"/gemm_*.log | wc -l)/$(ls "$O"/gemm_*.log | wc -l)"
touch "$O/DONE"
