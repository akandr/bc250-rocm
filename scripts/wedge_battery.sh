#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# What is still alive after the fault chain, with amdgpu.gpu_recovery=0 and the faulting process
# unkillable. Every step is timeout-wrapped: the point is to find where it stops, not to hang here too.
set -u
O=~/wedge-2026-09-22; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
say () { echo "=== $*" | tee -a "$O/battery.txt"; }
run () { local n=$1; shift; say "$n"; timeout -k 5 "$1" "${@:2}" >> "$O/battery.txt" 2>&1
         echo "    rc=$?" | tee -a "$O/battery.txt"; }

say "time $(date +%H:%M:%S), uptime $(uptime | sed 's/.*up //;s/,.*user.*//')"
say "faulting process: $(ps -eo pid,stat,comm,etime | grep -E '^ *683425' || echo gone)"
say "kfd holders: $(sudo fuser -v /dev/kfd 2>&1 | tail -1)"
say "fault lines this boot: $(sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE 'page fault \(src_id|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin')"
say "GPU reset lines: $(sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ci 'GPU reset begin')"
say "device still on the bus: $(lspci -s 01:00.0 | cut -c1-70)"

run "1. KFD enumeration (rocminfo, gfx1013 line)" 60 bash -c "LD_LIBRARY_PATH=$L rocminfo 2>&1 | grep -iE 'gfx1013|Compute Unit|Marketing' | head -6"
run "2. HIP device query + memory" 60 bash -c "LD_LIBRARY_PATH=$L ~/chain13/dispatch_floor 200 1 64 2>&1 | head -3"
run "3. Vulkan, the same board, a real model" 900 bash -c "~/llama-master/build-vk-f44/bin/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -mmp 0 -ngl 99 -fa on -p 0 -n 32 -r 2 2>&1 | grep -aE 'tg32|error'"
run "4. ROCm, a trivial correctness op" 300 bash -c "LD_LIBRARY_PATH=$L ~/llama-master/build-hip-pkf16/bin/test-backend-ops test -o ADD -b ROCm0 2>&1 | tail -3"
run "5. ROCm, the same model Vulkan just ran" 600 bash -c "LD_LIBRARY_PATH=$L ~/llama-master/build-hip-pkf16/bin/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -mmp 0 -ngl 99 -fa on -p 0 -n 32 -r 1 2>&1 | grep -aE 'tg32|error|abort' | head -3"

say "fault lines after the battery: $(sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE 'page fault \(src_id|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin')"
say "done $(date +%H:%M:%S)"
