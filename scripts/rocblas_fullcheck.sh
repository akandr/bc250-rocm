#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Full-output rocBLAS correctness battery (patches/rocblas_fullcheck.cpp): fp32, fp64, fp16 and
# fp16-in/fp32-compute GEMM, A normal and transposed, sizes 256 to 8192 including two that are not
# tile multiples, then two sustained runs; every element of every iteration checked. The kernel log
# delta of the whole battery is saved raw (dmesg_delta.txt) so any message, whatever its wording, is
# visible; dmesg_summary.txt greps it for the fault and failure messages of both schedulers.
# Run: rocblas_fullcheck.sh <out_dir> [binary]
set -u
O=${1:?out dir}; BIN=${2:-$HOME/fullcheck/rocblas_fullcheck}
mkdir -p "$O"; L=/opt/bc250-rocm/lib64:/usr/lib64
{
  echo "date: $(date -Is)"; echo "kernel: $(uname -r)"; echo "cmdline: $(cat /proc/cmdline)"
  for p in bc250_flush_pasid_kiq bc250_flush_by_runlist bc250_cc_write_mode sched_policy gpu_recovery; do
    printf 'amdgpu.%s=%s\n' "$p" "$(cat /sys/module/amdgpu/parameters/$p 2>/dev/null)"; done
  echo "simd: $(sudo -n cat /sys/kernel/debug/dri/0/amdgpu_gca_config 2>/dev/null | grep -m1 -i simd_count || echo n/a)"
  echo "sclk levels:"; cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null
  echo "rocblas: $(LD_LIBRARY_PATH=$L ldd "$BIN" | grep rocblas)"
  echo "HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-unset}"
} > "$O/header.txt"
sudo -n dmesg > "$O/dmesg_before.txt"
run() { # type op N iters
  echo "=== $*" >> "$O/battery.log"
  LD_LIBRARY_PATH=$L timeout -k 10 900 "$BIN" "$@" >> "$O/battery.log" 2>&1
  echo "rc=$? sclk_now=$(grep '\*' /sys/class/drm/card*/device/pp_dpm_sclk | tr -d '\n')" >> "$O/battery.log"
}
for c in skip wrong; do for t in s h; do   # negative controls: these runs must FAIL
  echo "=== control $c $t tn 1000 2" >> "$O/battery.log"
  FULLCHECK_CONTROL=$c LD_LIBRARY_PATH=$L timeout -k 10 300 "$BIN" $t tn 1000 2 >> "$O/battery.log" 2>&1
  echo "rc=$?" >> "$O/battery.log"
done; done
for t in s d h hs; do for op in nn tn; do for n in 256 512 1000 1024 2048 3001 4096 8192; do
  run $t $op $n 5
done; done; done
run s nn 4096 200
run hs tn 4096 200
sudo -n dmesg > "$O/dmesg_after.txt"
diff <(cat "$O/dmesg_before.txt") "$O/dmesg_after.txt" | grep '^>' | cut -c3- > "$O/dmesg_delta.txt"
grep -i -E 'page fault|preempt|evict|unrecoverable|hang|timeout|time out|reset|ring .* test failed|amdgpu.*error|kfd.*error|MEMORY_APERTURE' \
  "$O/dmesg_delta.txt" > "$O/dmesg_summary.txt"
{ echo "dmesg delta lines: $(wc -l < "$O/dmesg_delta.txt")"; echo "fault/failure lines: $(wc -l < "$O/dmesg_summary.txt")"
  grep -c ALL_ELEMENTS_CORRECT "$O/battery.log" | sed 's/^/runs all correct: /'
  grep -c ' FAIL$' "$O/battery.log" | sed 's/^/runs failed (the 4 negative controls must be among them): /'
  grep -c '^=== ' "$O/battery.log" | sed 's/^/runs started: /'; } > "$O/result.txt"
touch "$O/DONE"
